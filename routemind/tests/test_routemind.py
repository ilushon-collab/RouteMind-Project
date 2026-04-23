import os
import shutil
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace


TEST_AUTH_DIR = Path(__file__).resolve().parent / ".test_auth_store"
shutil.rmtree(TEST_AUTH_DIR, ignore_errors=True)
TEST_AUTH_DIR.mkdir(parents=True, exist_ok=True)
os.environ["ROUTEMIND_AUTH_DIR"] = str(TEST_AUTH_DIR)
os.environ["ROUTEMIND_SECRET_KEY"] = "routemind-test-secret"

from fastapi import HTTPException

from app import main
from app.auth import reset_auth_caches
from app.models import (
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
main.startup()


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

    def make_route(self, max_shift_time: float = 25) -> RouteRequest:
        return RouteRequest(
            depot=Depot(x=0, y=0),
            stops=[
                Stop(id=1, x=2, y=3, window_start=2, window_end=10, service_time=1, priority=3),
                Stop(id=2, x=5, y=2, window_start=0, window_end=8, service_time=1, priority=5),
                Stop(id=3, x=6, y=6, window_start=5, window_end=15, service_time=1, priority=2),
            ],
            max_shift_time=max_shift_time,
            weights=Weights(w_dist=1.0, w_wait=0.5, w_late=3.0, w_priority=2.0, w_shift=4.0),
            optimization=OptimizationConfig(algorithm="2opt", max_iterations=100, no_improvement_limit=10),
        )

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


if __name__ == "__main__":
    unittest.main()
