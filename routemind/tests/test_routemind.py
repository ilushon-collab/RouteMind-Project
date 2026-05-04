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

    # ------------------------------------------------------------------
    # Window & service-time validators
    # ------------------------------------------------------------------

    def test_window_end_less_than_window_start_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            Stop(id=1, x=0, y=0, window_start=10, window_end=5, service_time=1, priority=1)

    def test_service_time_negative_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            Stop(id=1, x=0, y=0, window_start=0, window_end=10, service_time=-1, priority=1)

    # ------------------------------------------------------------------
    # Scenario upsert
    # ------------------------------------------------------------------

    def test_scenario_with_same_name_is_updated_not_duplicated(self) -> None:
        user = self.make_user("upsert-test")
        route_3stops = self.make_route()
        route_1stop = RouteRequest(
            depot=Depot(x=0, y=0),
            stops=[Stop(id=1, x=1, y=1, window_start=0, window_end=10, service_time=1, priority=1)],
            max_shift_time=25,
            weights=Weights(),
        )

        s1 = main.create_or_update_scenario(
            ScenarioCreateRequest(name="My Route", route=route_3stops),
            DummyRequest("upsert-test"),
            user,
        )
        s2 = main.create_or_update_scenario(
            ScenarioCreateRequest(name="My Route", route=route_1stop),
            DummyRequest("upsert-test"),
            user,
        )

        # Same record updated, not a new one inserted
        self.assertEqual(s1.id, s2.id)
        self.assertEqual(s2.stop_count, 1)
        # Only one scenario in the list
        self.assertEqual(len(main.get_saved_scenarios(user)), 1)

    # ------------------------------------------------------------------
    # Optimize rate-limit enforcement
    # ------------------------------------------------------------------

    def test_optimize_rate_limit_is_enforced(self) -> None:
        user = self.make_user("opt-rl")
        route = RouteRequest(
            depot=Depot(x=0, y=0),
            stops=[Stop(id=1, x=1, y=1, window_start=0, window_end=10, service_time=1, priority=1)],
            max_shift_time=25,
            weights=Weights(),
            optimization=OptimizationConfig(max_iterations=5, no_improvement_limit=3),
        )

        for _ in range(30):
            main.optimize_route(route, DummyRequest("opt-rl"), None, user)

        with self.assertRaises(HTTPException) as ctx:
            main.optimize_route(route, DummyRequest("opt-rl"), None, user)

        self.assertEqual(ctx.exception.status_code, 429)

    # ------------------------------------------------------------------
    # Geo-based routes: realistic travel time & walking vs driving
    # ------------------------------------------------------------------

    def make_israel_route(self, travel_mode: str = "driving") -> RouteRequest:
        """Route with real geocoded coordinates for Israeli cities.

        Depot: Netanya city centre  (lat 32.33, lng 34.86)
        Stop 1: Ra'anana            (lat 32.16, lng 34.84) — ~19 km south on Route 4
        Stop 2: Hadera              (lat 32.44, lng 34.92) — ~13 km north-east on Highway 2

        All coordinates are real; the route is the canonical example used to
        verify that RouteMind produces realistic travel-time metrics.
        """
        return RouteRequest(
            depot=Depot(x=34.86, y=32.33, lat=32.33, lng=34.86),
            stops=[
                Stop(
                    id=1, x=34.84, y=32.16, lat=32.16, lng=34.84,
                    window_start=0, window_end=600, service_time=10, priority=3,
                ),
                Stop(
                    id=2, x=34.92, y=32.44, lat=32.44, lng=34.92,
                    window_start=0, window_end=600, service_time=10, priority=2,
                ),
            ],
            max_shift_time=480,
            weights=Weights(w_dist=1.0, w_wait=0.5, w_late=3.0, w_priority=2.0, w_shift=4.0),
            optimization=OptimizationConfig(algorithm="2opt", max_iterations=50, no_improvement_limit=10),
            travel_mode=travel_mode,
        )

    def test_geo_route_driving_produces_realistic_travel_time(self) -> None:
        """Netanya → Ra'anana + Ra'anana → Hadera + return should be
        roughly 60–65 km at 50 km/h ≈ 75–80 min driving, well under the
        480-minute shift window.
        """
        user = self.make_user("geo-driving")
        result = main.optimize_route(self.make_israel_route("driving"), DummyRequest("geo-driving"), None, user)

        # The three legs together span ~65 km; at 50 km/h that is ~78 min.
        # Allow a generous band to account for the optimiser choosing different
        # orderings (longest possible round-trip is still well under 180 min).
        self.assertGreater(result.total_travel_time, 30.0)
        self.assertLess(result.total_travel_time, 180.0)
        self.assertTrue(result.feasible)

    def test_walking_travel_time_is_ten_times_driving(self) -> None:
        """TravelTimeProvider uses 50 km/h for driving and 5 km/h for walking,
        so walking metrics must be exactly 10 × the driving metrics.
        """
        user = self.make_user("mode-ratio")

        driving_result = main.optimize_route(
            self.make_israel_route("driving"), DummyRequest("mode-ratio"), None, user
        )
        walking_result = main.optimize_route(
            self.make_israel_route("walking"), DummyRequest("mode-ratio"), None, user
        )

        self.assertGreater(walking_result.total_travel_time, driving_result.total_travel_time)
        ratio = walking_result.total_travel_time / driving_result.total_travel_time
        # Speed ratio is 50/5 = 10; route geometry is fixed so travel-time ratio must be 10.
        self.assertAlmostEqual(ratio, 10.0, delta=0.1)

    def test_travel_mode_is_persisted_in_history(self) -> None:
        """travel_mode must survive the JSON round-trip into optimization_runs."""
        user = self.make_user("mode-history")
        main.optimize_route(self.make_israel_route("walking"), DummyRequest("mode-history"), None, user)

        history = main.optimization_history(5, user)
        detail  = main.optimization_history_detail(history[0].id, user)
        self.assertEqual(detail.request.travel_mode, "walking")

    # ------------------------------------------------------------------
    # Cross-user data isolation
    # ------------------------------------------------------------------

    def test_scenario_cannot_be_read_by_a_different_user(self) -> None:
        user_a = self.make_user("iso-sc-a")
        user_b = self.make_user("iso-sc-b")
        route  = self.make_route()

        scenario = main.create_or_update_scenario(
            ScenarioCreateRequest(name="Private", route=route),
            DummyRequest("iso-sc-a"),
            user_a,
        )

        with self.assertRaises(HTTPException) as ctx:
            main.read_scenario(scenario.id, user_b)
        self.assertEqual(ctx.exception.status_code, 404)

        # User B's own list must be empty
        self.assertEqual(len(main.get_saved_scenarios(user_b)), 0)

    def test_history_cannot_be_read_by_a_different_user(self) -> None:
        user_a = self.make_user("iso-hi-a")
        user_b = self.make_user("iso-hi-b")
        route  = self.make_route()

        main.optimize_route(route, DummyRequest("iso-hi-a"), None, user_a)
        runs_a = main.optimization_history(10, user_a)
        self.assertEqual(len(runs_a), 1)

        with self.assertRaises(HTTPException) as ctx:
            main.optimization_history_detail(runs_a[0].id, user_b)
        self.assertEqual(ctx.exception.status_code, 404)

        # User B's own history must be empty
        self.assertEqual(len(main.optimization_history(10, user_b)), 0)


class HttpAuthFlowTests(unittest.TestCase):
    """
    End-to-end HTTP-level tests that exercise the full FastAPI stack
    (CORS middleware, request body parsing, response serialisation).
    These complement the unit tests above which bypass the HTTP layer.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from starlette.testclient import TestClient  # noqa: PLC0415
        from app.auth import init_auth_storage, reset_auth_caches  # noqa: PLC0415
        from app.storage import init_app_storage  # noqa: PLC0415

        # The previous test class deletes TEST_AUTH_DIR in tearDownClass.
        # Recreate the directory and DB schema so HTTP tests have a valid database.
        TEST_AUTH_DIR.mkdir(parents=True, exist_ok=True)
        reset_auth_caches()
        init_auth_storage()
        init_app_storage()

        # Reset in-memory rate limiter so unit-test registrations don't bleed over.
        main.rate_limiter.reset()
        cls.client = TestClient(main.app, raise_server_exceptions=False)

    def setUp(self) -> None:
        # Clear per-test so tests within this class don't exhaust each other's limits.
        main.rate_limiter.reset()

    def _unique(self) -> str:
        return uuid.uuid4().hex[:10]

    # ------------------------------------------------------------------
    # POST /register
    # ------------------------------------------------------------------

    def test_http_register_returns_201_with_token_and_user(self) -> None:
        suffix = self._unique()
        resp = self.client.post(
            "/register",
            json={"username": f"http{suffix}", "email": f"http{suffix}@test.com", "password": "password123"},
        )
        self.assertEqual(resp.status_code, 201)
        body = resp.json()
        self.assertIn("access_token", body)
        self.assertIsInstance(body["access_token"], str)
        self.assertGreater(len(body["access_token"]), 10)
        self.assertEqual(body["user"]["username"], f"http{suffix}")
        self.assertEqual(body["user"]["email"], f"http{suffix}@test.com")
        self.assertIn("id", body["user"])
        self.assertIn("created_at", body["user"])

    def test_http_register_duplicate_username_returns_400(self) -> None:
        suffix = self._unique()
        payload = {"username": f"dup{suffix}", "email": f"dup{suffix}@test.com", "password": "password123"}
        self.client.post("/register", json=payload)
        resp = self.client.post("/register", json={**payload, "email": f"other{suffix}@test.com"})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("detail", resp.json())

    def test_http_register_duplicate_email_returns_400(self) -> None:
        suffix = self._unique()
        payload = {"username": f"em{suffix}", "email": f"em{suffix}@test.com", "password": "password123"}
        self.client.post("/register", json=payload)
        resp = self.client.post("/register", json={**payload, "username": f"em2{suffix}"})
        self.assertEqual(resp.status_code, 400)

    def test_http_register_short_password_returns_422(self) -> None:
        suffix = self._unique()
        resp = self.client.post(
            "/register",
            json={"username": f"pw{suffix}", "email": f"pw{suffix}@test.com", "password": "short"},
        )
        self.assertEqual(resp.status_code, 422)
        detail = resp.json()["detail"]
        self.assertTrue(
            any("8" in (item.get("msg") or "") for item in detail),
            msg=f"Expected min-length message, got: {detail}",
        )

    def test_http_register_missing_field_returns_422(self) -> None:
        resp = self.client.post("/register", json={"username": "nopassword", "email": "np@test.com"})
        self.assertEqual(resp.status_code, 422)

    def test_http_register_get_serves_html(self) -> None:
        resp = self.client.get("/register")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/html", resp.headers["content-type"])

    # ------------------------------------------------------------------
    # POST /login
    # ------------------------------------------------------------------

    def test_http_login_with_username_returns_200_with_token(self) -> None:
        suffix = self._unique()
        self.client.post(
            "/register",
            json={"username": f"lg{suffix}", "email": f"lg{suffix}@test.com", "password": "password123"},
        )
        resp = self.client.post("/login", json={"identity": f"lg{suffix}", "password": "password123"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("access_token", body)
        self.assertEqual(body["user"]["username"], f"lg{suffix}")

    def test_http_login_with_email_returns_200_with_token(self) -> None:
        suffix = self._unique()
        self.client.post(
            "/register",
            json={"username": f"le{suffix}", "email": f"le{suffix}@test.com", "password": "password123"},
        )
        resp = self.client.post("/login", json={"identity": f"le{suffix}@test.com", "password": "password123"})
        self.assertEqual(resp.status_code, 200)

    def test_http_login_wrong_password_returns_401(self) -> None:
        suffix = self._unique()
        self.client.post(
            "/register",
            json={"username": f"lw{suffix}", "email": f"lw{suffix}@test.com", "password": "password123"},
        )
        resp = self.client.post("/login", json={"identity": f"lw{suffix}", "password": "wrongpassword"})
        self.assertEqual(resp.status_code, 401)

    def test_http_login_unknown_user_returns_401(self) -> None:
        resp = self.client.post("/login", json={"identity": "nobody_here", "password": "password123"})
        self.assertEqual(resp.status_code, 401)

    # ------------------------------------------------------------------
    # GET /me
    # ------------------------------------------------------------------

    def test_http_me_with_valid_token_returns_user(self) -> None:
        suffix = self._unique()
        reg = self.client.post(
            "/register",
            json={"username": f"me{suffix}", "email": f"me{suffix}@test.com", "password": "password123"},
        )
        token = reg.json()["access_token"]
        resp = self.client.get("/me", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["username"], f"me{suffix}")

    def test_http_me_without_token_returns_401(self) -> None:
        resp = self.client.get("/me")
        self.assertEqual(resp.status_code, 401)

    # ------------------------------------------------------------------
    # CORS — cross-origin preflight and actual POST
    # ------------------------------------------------------------------

    def test_http_cors_preflight_for_register_returns_200(self) -> None:
        resp = self.client.options(
            "/register",
            headers={
                "Origin": "http://192.168.1.100:8000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("access-control-allow-origin", resp.headers)

    def test_http_cors_cross_origin_post_includes_acao_header(self) -> None:
        suffix = self._unique()
        resp = self.client.post(
            "/register",
            json={"username": f"co{suffix}", "email": f"co{suffix}@test.com", "password": "password123"},
            headers={"Origin": "http://192.168.1.100:8000"},
        )
        self.assertEqual(resp.status_code, 201)
        self.assertIn("access-control-allow-origin", resp.headers)

    # ------------------------------------------------------------------
    # HTML serving — Cache-Control header
    # ------------------------------------------------------------------

    def test_http_root_serves_html_with_no_store(self) -> None:
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/html", resp.headers["content-type"])
        self.assertIn("no-store", resp.headers.get("cache-control", ""))

    def test_http_register_get_serves_html_with_no_store(self) -> None:
        resp = self.client.get("/register")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("no-store", resp.headers.get("cache-control", ""))

    def test_http_login_get_serves_html_with_no_store(self) -> None:
        resp = self.client.get("/login")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("no-store", resp.headers.get("cache-control", ""))


if __name__ == "__main__":
    unittest.main()
