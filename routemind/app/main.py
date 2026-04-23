import logging
import time
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
    LoginRequest,
    OptimizationConfig,
    OptimizationHistoryDetail,
    OptimizationHistorySummary,
    RegisterRequest,
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
from app.utils import build_distance_matrix


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("routemind")

app = FastAPI(title="RouteMind API", version="3.0")
bearer_scheme = HTTPBearer(auto_error=False)
rate_limiter = InMemoryRateLimiter()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.on_event("startup")
def startup() -> None:
    init_auth_storage()
    init_app_storage()


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
    return FileResponse(static_dir / "index.html")


@app.post("/optimize", response_model=RouteResponse)
def optimize_route(
    data: RouteRequest,
    request: Request,
    scenario_id: int | None = Query(default=None),
    current_user: dict = Depends(get_current_user),
) -> RouteResponse:
    enforce_rate_limit(request, "optimize", limit=30, window_seconds=60, current_user=current_user)

    try:
        distance_matrix = build_distance_matrix(data.depot, data.stops)

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
    return FileResponse(static_dir / "index.html")
