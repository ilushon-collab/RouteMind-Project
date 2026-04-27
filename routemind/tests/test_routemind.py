import os
import shutil
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

from fastapi import HTTPException
from pydantic import ValidationError


TEST_AUTH_DIR = Path(__file__).resolve().parent / ".test_auth_store"
shutil.rmtree(TEST_AUTH_DIR, ignore_errors=True)
TEST_AUTH_DIR.mkdir(parents=True, exist_ok=True)
os.environ["ROUTEMIND_AUTH_DIR"] = str(TEST_AUTH_DIR)
os.environ["ROUTEMIND_SECRET_KEY"] = "routemind-test-secret"

from app import main  # noqa: E402
from app.auth import reset_auth_caches  # noqa: E402
from app.models import (  # noqa: E402
    LoginRequest,
    OptimizationConfig,
    RegisterRequest,
    RouteRequest,
    ScenarioCreateRequest,
    Stop,
    Depot,
    Weights,
)


reset_auth_caches()
from app.auth import init_auth_storage  # noqa: E402
from app.storage import init_app_storage  # noqa: E402
init_auth_storage()
init_app_storage()


class DummyRequest:
    def __init__(self, host: str = "127.0.0.1") -> None:
        self.client = SimpleNamespace(host=host)
        self.method = "POST"
        self.url = SimpleNamespace(path="/test")


class RouteMindTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(TEST_AUTH_DIR, ignore_errors=True)

    def make_user(self, host: str = "127.0.0.1") -> dict:
        suffix = uuid.uuid4().hex[:10]
        auth = main.register(
            DummyRequest(host),
            RegisterRequest(
                username=f"tester{suffix}",
                email=f"tester{suffix}@example.com",
                password="password123",
            ),
        )
        return auth.user.model_dump(mode="json")

    def make_route(self, max_shift_time: float = 25, algorithm: str = "2opt") -> RouteRequest:
        return RouteRequest(
            depot=Depot(x=0, y=0),
            stops=[
                Stop(id=1, x=2, y=3, window_start=2, window_end=10, service_time=1, priority=3),
                Stop(id=2, x=5, y=2, window_start=0, window_end=8, service_time=1, priority=5),
                Stop(id=3, x=6, y=6, window_start=5, window_end=15, service_time=1, priority=2),
            ],
            max_shift_time=max_shift_time,
            weights=Weights(w_dist=1.0, w_wait=0.5, w_late=3.0, w_priority=2.0, w_shift=4.0),
            optimization=OptimizationConfig(algorithm=algorithm, max_iterations=100, no_improvement_limit=10),
        )

    # ------------------------------------------------------------------
    # Original tests
    # ------------------------------------------------------------------

    def test_login_and_optimize_endpoint_is_protected(self) -> None:
        user = self.make_user("auth-check")
        login_response = main.login(
            DummyRequest("auth-check"),
            LoginRequest(identity=user["email"], password="password123"),
        )

        self.assertEqual(login_response.user.email, user["email"])

        optimize_route = next(
            route for route in main.app.routes if getattr(route, "path", None) == "/optimize" and "POST" in route.methods
        )
        dependency_calls = {dependency.call for dependency in optimize_route.dependant.dependencies}
        self.assertIn(main.get_current_user, dependency_calls)

    def test_scenario_persistence_and_history(self) -> None:
        user = self.make_user("scenario-flow")
        route = self.make_route()

        scenario = main.create_or_update_scenario(
            ScenarioCreateRequest(name="Morning Run", route=route),
            DummyRequest("scenario-flow"),
            user,
        )
        self.assertEqual(scenario.name, "Morning Run")

        scenarios = main.get_saved_scenarios(user)
        self.assertEqual(len(scenarios), 1)
        self.assertEqual(scenarios[0].stop_count, 3)

        loaded = main.read_scenario(scenario.id, user)
        self.assertEqual(loaded.route.stops[0].id, 1)

        result = main.optimize_route(route, DummyRequest("scenario-flow"), scenario.id, user)
        self.assertGreater(result.optimized_cost, 0)

        history = main.optimization_history(20, user)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].scenario_id, scenario.id)

        detail = main.optimization_history_detail(history[0].id, user)
        self.assertEqual(detail.scenario_name, "Morning Run")
        self.assertEqual(len(detail.response.optimized_route_order), 3)

    def test_infeasible_route_keeps_all_stops_and_reports_overrun(self) -> None:
        user = self.make_user("infeasible-route")
        route = self.make_route(max_shift_time=5)

        result = main.optimize_route(route, DummyRequest("infeasible-route"), None, user)

        self.assertEqual(len(result.initial_route_order), len(route.stops))
        self.assertEqual(len(result.optimized_route_order), len(route.stops))
        self.assertFalse(result.feasible)
        self.assertGreater(result.shift_overrun, 0)
        self.assertGreaterEqual(result.priority_adjusted_lateness, result.total_lateness)

    def test_register_rate_limit_is_enforced(self) -> None:
        for idx in range(5):
            suffix = uuid.uuid4().hex[:10]
            response = main.register(
                DummyRequest("rate-limit"),
                RegisterRequest(
                    username=f"limited{idx}{suffix}",
                    email=f"limited{idx}{suffix}@example.com",
                    password="password123",
                ),
            )
            self.assertTrue(response.access_token)

        with self.assertRaises(HTTPException) as error:
            suffix = uuid.uuid4().hex[:10]
            main.register(
                DummyRequest("rate-limit"),
                RegisterRequest(
                    username=f"limited-final-{suffix}",
                    email=f"limited-final-{suffix}@example.com",
                    password="password123",
                ),
            )

        self.assertEqual(error.exception.status_code, 429)

    # ------------------------------------------------------------------
    # Swap algorithm
    # ------------------------------------------------------------------

    def test_swap_algorithm_produces_valid_result(self) -> None:
        user = self.make_user("swap-algo")
        route = self.make_route(algorithm="swap")

        result = main.optimize_route(route, DummyRequest("swap-algo"), None, user)

        self.assertEqual(result.algorithm_used, "swap")
        self.assertGreater(result.optimized_cost, 0)
        self.assertEqual(len(result.optimized_route_order), len(route.stops))

    # ------------------------------------------------------------------
    # Relocate algorithm
    # ------------------------------------------------------------------

    def test_relocate_algorithm_produces_valid_result(self) -> None:
        user = self.make_user("relocate-algo")
        route = self.make_route(algorithm="relocate")

        result = main.optimize_route(route, DummyRequest("relocate-algo"), None, user)

        self.assertEqual(result.algorithm_used, "relocate")
        self.assertGreater(result.optimized_cost, 0)
        self.assertEqual(len(result.optimized_route_order), len(route.stops))

    # ------------------------------------------------------------------
    # GET /me
    # ------------------------------------------------------------------

    def test_me_endpoint_returns_current_user(self) -> None:
        user = self.make_user("me-check")

        result = main.me(user)

        self.assertEqual(result.id, user["id"])
        self.assertEqual(result.username, user["username"])
        self.assertEqual(result.email, user["email"])

    # ------------------------------------------------------------------
    # GET /history/{run_id}
    # ------------------------------------------------------------------

    def test_history_detail_returns_full_run(self) -> None:
        user = self.make_user("hist-detail")
        route = self.make_route()

        main.optimize_route(route, DummyRequest("hist-detail"), None, user)
        history = main.optimization_history(20, user)
        self.assertEqual(len(history), 1)

        detail = main.optimization_history_detail(history[0].id, user)

        self.assertEqual(len(detail.response.optimized_route_order), len(route.stops))
        self.assertEqual(detail.request.max_shift_time, route.max_shift_time)
        self.assertEqual(detail.algorithm_used, "2opt")

    def test_history_detail_not_found_returns_404(self) -> None:
        user = self.make_user("hist-404")

        with self.assertRaises(HTTPException) as ctx:
            main.optimization_history_detail(999999, user)

        self.assertEqual(ctx.exception.status_code, 404)

    # ------------------------------------------------------------------
    # Duplicate stop IDs validation
    # ------------------------------------------------------------------

    def test_duplicate_stop_ids_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            RouteRequest(
                depot=Depot(x=0, y=0),
                stops=[
                    Stop(id=1, x=2, y=3, window_start=2, window_end=10, service_time=1, priority=3),
                    Stop(id=1, x=5, y=2, window_start=0, window_end=8, service_time=1, priority=5),
                ],
                max_shift_time=25,
                weights=Weights(),
            )

    # ------------------------------------------------------------------
    # max_shift_time <= 0 validation
    # ------------------------------------------------------------------

    def test_max_shift_time_zero_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            RouteRequest(
                depot=Depot(x=0, y=0),
                stops=[Stop(id=1, x=2, y=3, window_start=2, window_end=10, service_time=1, priority=3)],
                max_shift_time=0,
                weights=Weights(),
            )

    def test_max_shift_time_negative_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            RouteRequest(
                depot=Depot(x=0, y=0),
                stops=[Stop(id=1, x=2, y=3, window_start=2, window_end=10, service_time=1, priority=3)],
                max_shift_time=-5,
                weights=Weights(),
            )

    # ------------------------------------------------------------------
    # Login rate limiting
    # ------------------------------------------------------------------

    def test_login_rate_limit_is_enforced(self) -> None:
        suffix = uuid.uuid4().hex[:10]
        host = f"login-rl-{suffix}"
        main.register(
            DummyRequest(host),
            RegisterRequest(
                username=f"lrl{suffix}",
                email=f"lrl{suffix}@example.com",
                password="password123",
            ),
        )

        for _ in range(10):
            main.login(
                DummyRequest(host),
                LoginRequest(identity=f"lrl{suffix}@example.com", password="password123"),
            )

        with self.assertRaises(HTTPException) as ctx:
            main.login(
                DummyRequest(host),
                LoginRequest(identity=f"lrl{suffix}@example.com", password="password123"),
            )

        self.assertEqual(ctx.exception.status_code, 429)

    # ------------------------------------------------------------------
    # DELETE /scenarios/{id}
    # ------------------------------------------------------------------

    def test_delete_scenario_removes_it(self) -> None:
        user = self.make_user("del-scenario")
        route = self.make_route()
        scenario = main.create_or_update_scenario(
            ScenarioCreateRequest(name="To Delete", route=route),
            DummyRequest("del-scenario"),
            user,
        )

        main.remove_scenario(scenario.id, user)

        remaining = main.get_saved_scenarios(user)
        self.assertEqual(len(remaining), 0)

    def test_delete_nonexistent_scenario_returns_404(self) -> None:
        user = self.make_user("del-404")

        with self.assertRaises(HTTPException) as ctx:
            main.remove_scenario(999999, user)

        self.assertEqual(ctx.exception.status_code, 404)

    # ------------------------------------------------------------------
    # Scenario 404 for GET
    # ------------------------------------------------------------------

    def test_read_nonexistent_scenario_returns_404(self) -> None:
        user = self.make_user("no-scenario")

        with self.assertRaises(HTTPException) as ctx:
            main.read_scenario(999999, user)

        self.assertEqual(ctx.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
