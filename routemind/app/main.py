import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles

from app.auth import (
    authenticate_user,
    create_access_token,
    create_user,
    decode_access_token,
    get_user_by_id,
    init_auth_storage,
)
from app.evaluator import evaluate_route
from app.models import (
    AuthResponse,
    GeocodeResult,
    LoginRequest,
    OptimizationConfig,
    OptimizationHistoryDetail,
    OptimizationHistorySummary,
    RegisterRequest,
    RoadRouteRequest,
    RoadWaypoint,
    RouteRequest,
    RouteResponse,
    ScenarioCreateRequest,
    ScenarioDetail,
    ScenarioSummary,
    UserResponse,
)
from app.optimizer import (
    greedy_initial_route,
    improve_route_2opt,
    local_search_relocate,
    local_search_swap,
)
from app.rate_limit import InMemoryRateLimiter
from app.storage import (
    delete_scenario,
    get_optimization_run,
    get_scenario,
    init_app_storage,
    list_optimization_runs,
    list_scenarios,
    record_optimization_run,
    save_scenario,
)
from app.utils import HaversineDistanceProvider, build_distance_matrix


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("routemind")


@asynccontextmanager
async def lifespan(application: FastAPI):
    init_auth_storage()
    init_app_storage()
    yield


app = FastAPI(title="RouteMind API", version="3.0", lifespan=lifespan)
bearer_scheme = HTTPBearer(auto_error=False)
rate_limiter = InMemoryRateLimiter()
haversine_provider = HaversineDistanceProvider()
# Route-bend constants are coordinate-degree deltas used only for drawing the
# browser polyline; timing/distance still come from geographic distance.
MIN_ROUTE_BEND_DELTA = 0.002
MAX_ROUTE_BEND_DELTA = 0.02
ROUTE_BEND_SCALE = 0.18

app.add_middleware(
    CORSMiddleware,
    # allow_origins=["*"] lets browsers on any origin (LAN IP, custom hostname, etc.)
    # reach the API when running self-hosted.  The app authenticates every sensitive
    # endpoint with Bearer tokens, so there is no cookie-based CSRF exposure.
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    started_at = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        logger.exception("request_failed method=%s path=%s duration_ms=%.2f", request.method, request.url.path, elapsed_ms)
        raise

    elapsed_ms = (time.perf_counter() - started_at) * 1000
    logger.info(
        "request_completed method=%s path=%s status=%s duration_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> dict:
    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required.")

    try:
        payload = decode_access_token(credentials.credentials)
        user_id = int(payload["sub"])
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token.") from None

    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found.")

    return user


def client_identity(request: Request, current_user: dict | None = None) -> str:
    if current_user:
        return f"user:{current_user['id']}"
    if request.client and request.client.host:
        return f"ip:{request.client.host}"
    return "anonymous"


def enforce_rate_limit(
    request: Request,
    bucket: str,
    limit: int,
    window_seconds: int,
    current_user: dict | None = None,
) -> None:
    rate_limiter.check(f"{bucket}:{client_identity(request, current_user)}", limit, window_seconds)


@app.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def register(request: Request, data: RegisterRequest) -> AuthResponse:
    enforce_rate_limit(request, "register", limit=5, window_seconds=60)

    try:
        user = create_user(data.username, data.email, data.password)
    except ValueError as exc:
        logger.warning("register_failed username=%s reason=%s", data.username, str(exc))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    logger.info("register_success user_id=%s username=%s", user["id"], user["username"])
    return AuthResponse(
        access_token=create_access_token(user["id"]),
        user=UserResponse.model_validate(user),
    )


@app.post("/login", response_model=AuthResponse)
def login(request: Request, data: LoginRequest) -> AuthResponse:
    enforce_rate_limit(request, "login", limit=10, window_seconds=60)

    user = authenticate_user(data.identity, data.password)
    if not user:
        logger.warning("login_failed identity=%s", data.identity)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials.")

    logger.info("login_success user_id=%s username=%s", user["id"], user["username"])
    return AuthResponse(
        access_token=create_access_token(user["id"]),
        user=UserResponse.model_validate(user),
    )


@app.get("/me", response_model=UserResponse)
def me(current_user: dict = Depends(get_current_user)) -> UserResponse:
    return UserResponse.model_validate(current_user)


@app.get("/login")
@app.get("/register")
async def auth_entrypoint_fallback():
    return FileResponse(static_dir / "index.html", headers={"Cache-Control": "no-store"})


LOCAL_GEOCODER_PLACES = [
    {
        "lat": 51.5308,
        "lon": -0.1238,
        "display_name": "King's Cross Station, Euston Road, London, United Kingdom",
        "address": {"road": "King's Cross", "city": "London", "country_code": "gb"},
        "aliases": ["king's cross", "kings cross", "euston road", "station"],
    },
    {
        "lat": 51.5055,
        "lon": -0.0754,
        "display_name": "Tower Bridge, Tower Bridge Road, London, United Kingdom",
        "address": {"road": "Tower Bridge Road", "city": "London", "country_code": "gb"},
        "aliases": ["tower bridge", "tower bridge road"],
    },
    {
        "lat": 51.5027,
        "lon": -0.1528,
        "display_name": "Hyde Park Corner, London, United Kingdom",
        "address": {"road": "Hyde Park Corner", "city": "London", "country_code": "gb"},
        "aliases": ["hyde park", "hyde park corner"],
    },
    {
        "lat": 51.5054,
        "lon": -0.0235,
        "display_name": "Canary Wharf, London, United Kingdom",
        "address": {"road": "Canary Wharf", "city": "London", "country_code": "gb"},
        "aliases": ["canary wharf"],
    },
    {
        "lat": 40.7580,
        "lon": -73.9855,
        "display_name": "Times Square, Manhattan, New York, United States",
        "address": {"road": "Times Square", "city": "New York", "country_code": "us"},
        "aliases": ["times square", "manhattan"],
    },
    {
        "lat": 40.7484,
        "lon": -73.9857,
        "display_name": "Empire State Building, 20 W 34th Street, New York, United States",
        "address": {"road": "W 34th Street", "city": "New York", "country_code": "us"},
        "aliases": ["empire state", "34th street"],
    },
    {
        "lat": 48.8584,
        "lon": 2.2945,
        "display_name": "Eiffel Tower, Paris, France",
        "address": {"road": "Champ de Mars", "city": "Paris", "country_code": "fr"},
        "aliases": ["eiffel tower", "champ de mars"],
    },
]


def _place_matches(place: dict, query: str, country_code: str | None) -> bool:
    if country_code and place["address"].get("country_code") != country_code.lower():
        return False
    searchable = _normalize_place_text(" ".join([place["display_name"], *place["aliases"]]))
    query_parts = _normalize_place_text(query).split()
    return all(part in searchable for part in query_parts)


def _normalize_place_text(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else " " for char in value)


def _calculate_distance_km(a_lat: float, a_lng: float, b_lat: float, b_lng: float) -> float:
    return haversine_provider.distance(
        RoadWaypoint(lat=a_lat, lng=a_lng),
        RoadWaypoint(lat=b_lat, lng=b_lng),
    )


def _segment_geometry(start: tuple[float, float], end: tuple[float, float]) -> list[list[float]]:
    start_lat, start_lng = start
    end_lat, end_lng = end
    mid_lat = (start_lat + end_lat) / 2
    mid_lng = (start_lng + end_lng) / 2
    coordinate_delta = abs(end_lat - start_lat) + abs(end_lng - start_lng)
    # Keep curvature visible on short city trips while capping it so longer
    # routes do not arc unrealistically far away from their endpoints.
    bend = min(max(coordinate_delta, MIN_ROUTE_BEND_DELTA), MAX_ROUTE_BEND_DELTA) * ROUTE_BEND_SCALE
    return [
        [start_lat, start_lng],
        [start_lat + (mid_lat - start_lat) * 0.8, mid_lng - bend],
        [mid_lat + bend, mid_lng],
        [mid_lat + (end_lat - mid_lat) * 0.8, mid_lng + bend],
        [end_lat, end_lng],
    ]


@app.get("/geocode/search", response_model=list[GeocodeResult])
def local_geocode_search(
    q: str = Query(min_length=1),
    country_code: str | None = Query(default=None, min_length=2, max_length=2),
    limit: int = Query(default=6, ge=1, le=10),
) -> list[GeocodeResult]:
    matches = [
        GeocodeResult(
            lat=place["lat"],
            lon=place["lon"],
            display_name=place["display_name"],
            address=place["address"],
        )
        for place in LOCAL_GEOCODER_PLACES
        if _place_matches(place, q, country_code)
    ]
    return matches[:limit]


@app.post("/route-road")
def local_road_route(payload: RoadRouteRequest) -> dict:
    if len(payload.waypoints) < 2:
        raise HTTPException(status_code=422, detail="At least two waypoints are required.")

    speed_kmh = 34.0 if payload.mode == "driving" else 4.8
    road_factor = 1.28 if payload.mode == "driving" else 1.12
    lat_lngs: list[list[float]] = []
    legs: list[dict] = []
    total_distance = 0.0
    total_duration = 0.0

    for idx, (start, end) in enumerate(zip(payload.waypoints, payload.waypoints[1:])):
        km = _calculate_distance_km(start.lat, start.lng, end.lat, end.lng) * road_factor
        distance_m = km * 1000
        duration_s = (km / speed_kmh) * 3600
        segment_points = _segment_geometry((start.lat, start.lng), (end.lat, end.lng))
        lat_lngs.extend(segment_points if idx == 0 else segment_points[1:])
        total_distance += distance_m
        total_duration += duration_s
        legs.append(
            {
                "duration": duration_s,
                "distance": distance_m,
                "steps": [
                    {
                        "name": f"route segment {idx + 1}",
                        "distance": distance_m,
                        "duration": duration_s,
                        "maneuver": {"type": "continue", "modifier": "straight"},
                    },
                    {
                        "name": "",
                        "distance": 0,
                        "duration": 0,
                        "maneuver": {"type": "arrive"},
                    },
                ],
            }
        )

    return {
        "latLngs": lat_lngs,
        "duration": total_duration,
        "distance": total_distance,
        "legs": legs,
    }


@app.post("/optimize", response_model=RouteResponse)
def optimize_route(
    data: RouteRequest,
    request: Request,
    scenario_id: int | None = Query(default=None),
    current_user: dict = Depends(get_current_user),
) -> RouteResponse:
    enforce_rate_limit(request, "optimize", limit=30, window_seconds=60, current_user=current_user)

    try:
        distance_matrix = build_distance_matrix(data.depot, data.stops, travel_mode=data.travel_mode)

        initial_route = greedy_initial_route(
            data.depot,
            data.stops,
            data.max_shift_time,
            data.weights,
            distance_matrix=distance_matrix,
        )

        initial_eval = evaluate_route(
            data.depot,
            initial_route,
            data.max_shift_time,
            data.weights,
            distance_matrix=distance_matrix,
        )

        config = data.optimization or OptimizationConfig()

        algorithm_map = {
            "2opt": improve_route_2opt,
            "swap": local_search_swap,
            "relocate": local_search_relocate,
        }
        algorithm = algorithm_map.get(config.algorithm, improve_route_2opt)
        improved_route, improved_eval, opt_metadata = algorithm(
            data.depot,
            initial_route,
            data.max_shift_time,
            data.weights,
            max_iterations=config.max_iterations,
            no_improvement_limit=config.no_improvement_limit,
            time_limit=config.time_limit,
            distance_matrix=distance_matrix,
        )

        improvement_percent = 0.0
        if initial_eval["total_cost"] > 0:
            improvement_percent = (
                (initial_eval["total_cost"] - improved_eval["total_cost"]) / initial_eval["total_cost"]
            ) * 100

        response = RouteResponse(
            initial_route_order=[s.id for s in initial_route],
            optimized_route_order=[s.id for s in improved_route],
            initial_cost=initial_eval["total_cost"],
            optimized_cost=improved_eval["total_cost"],
            total_travel_time=improved_eval["total_travel_time"],
            total_wait_time=improved_eval["total_wait_time"],
            total_lateness=improved_eval["total_lateness"],
            priority_adjusted_lateness=improved_eval["priority_adjusted_lateness"],
            total_penalty=improved_eval["total_penalty"],
            distance_cost=improved_eval["distance_cost"],
            wait_cost=improved_eval["wait_cost"],
            lateness_cost=improved_eval["lateness_cost"],
            shift_overrun=improved_eval["shift_overrun"],
            shift_overrun_cost=improved_eval["shift_overrun_cost"],
            finish_time=improved_eval["finish_time"],
            feasible=improved_eval["feasible"],
            improvement_percent=improvement_percent,
            improvement_found=opt_metadata["improvement_found"],
            iterations_used=opt_metadata["iterations_used"],
            stopped_by=opt_metadata["stopped_by"],
            algorithm_used=config.algorithm,
            visits=improved_eval["visits"],
        )

        run_id = record_optimization_run(current_user["id"], data, response, scenario_id=scenario_id)
        logger.info(
            "optimize_success user_id=%s run_id=%s feasible=%s algorithm=%s",
            current_user["id"],
            run_id,
            response.feasible,
            response.algorithm_used,
        )
        return response
    except ValueError as exc:
        logger.warning("optimize_failed user_id=%s reason=%s", current_user["id"], str(exc))
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("optimize_error user_id=%s", current_user["id"])
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(exc)}") from exc


@app.get("/scenarios", response_model=list[ScenarioSummary])
def get_saved_scenarios(current_user: dict = Depends(get_current_user)) -> list[ScenarioSummary]:
    return list_scenarios(current_user["id"])


@app.post("/scenarios", response_model=ScenarioDetail, status_code=status.HTTP_201_CREATED)
def create_or_update_scenario(
    payload: ScenarioCreateRequest,
    request: Request,
    current_user: dict = Depends(get_current_user),
) -> ScenarioDetail:
    enforce_rate_limit(request, "scenarios.write", limit=20, window_seconds=60, current_user=current_user)
    scenario = save_scenario(current_user["id"], payload.name, payload.route)
    logger.info("scenario_saved user_id=%s scenario_id=%s", current_user["id"], scenario.id)
    return scenario


@app.get("/scenarios/{scenario_id}", response_model=ScenarioDetail)
def read_scenario(scenario_id: int, current_user: dict = Depends(get_current_user)) -> ScenarioDetail:
    scenario = get_scenario(current_user["id"], scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found.")
    return scenario


@app.delete("/scenarios/{scenario_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_scenario(scenario_id: int, current_user: dict = Depends(get_current_user)) -> None:
    deleted = delete_scenario(current_user["id"], scenario_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Scenario not found.")
    logger.info("scenario_deleted user_id=%s scenario_id=%s", current_user["id"], scenario_id)


@app.get("/history", response_model=list[OptimizationHistorySummary])
def optimization_history(
    limit: int = Query(default=20, ge=1, le=100),
    current_user: dict = Depends(get_current_user),
) -> list[OptimizationHistorySummary]:
    return [OptimizationHistorySummary.model_validate(item) for item in list_optimization_runs(current_user["id"], limit=limit)]


@app.get("/history/{run_id}", response_model=OptimizationHistoryDetail)
def optimization_history_detail(run_id: int, current_user: dict = Depends(get_current_user)) -> OptimizationHistoryDetail:
    item = get_optimization_run(current_user["id"], run_id)
    if not item:
        raise HTTPException(status_code=404, detail="Optimization run not found.")
    return item


@app.get("/health")
def health_check() -> dict:
    return {"status": "operational", "version": "3.0"}


@app.get("/")
async def root():
    return FileResponse(static_dir / "index.html", headers={"Cache-Control": "no-store"})
