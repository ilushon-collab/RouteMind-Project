from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Point(BaseModel):
    x: float
    y: float


class Stop(BaseModel):
    id: int
    x: float
    y: float
    lat: Optional[float] = None
    lng: Optional[float] = None
    label: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    street: Optional[str] = None
    house_number: Optional[str] = None
    window_start: float
    window_end: float
    service_time: float
    priority: int = Field(ge=1, le=5)

    @field_validator('service_time')
    @classmethod
    def validate_service_time(cls, v):
        if v < 0:
            raise ValueError('service_time must be >= 0')
        return v

    @field_validator('window_end')
    @classmethod
    def validate_window(cls, v, info):
        if 'window_start' in info.data and v < info.data['window_start']:
            raise ValueError('window_end must be >= window_start')
        return v


class Depot(BaseModel):
    x: float
    y: float
    lat: Optional[float] = None
    lng: Optional[float] = None
    label: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    street: Optional[str] = None
    house_number: Optional[str] = None


class Weights(BaseModel):
    w_dist: float = 1.0
    w_wait: float = 1.0
    w_late: float = 2.0
    w_priority: float = 2.0
    w_shift: float = 4.0


class OptimizationConfig(BaseModel):
    """Configuration for local search algorithm."""
    algorithm: Literal["2opt", "swap", "relocate"] = "2opt"
    max_iterations: int = Field(default=1000, ge=1)
    no_improvement_limit: int = Field(default=100, ge=1)
    time_limit: Optional[float] = Field(default=None, ge=0)


class RouteRequest(BaseModel):
    depot: Depot
    stops: List[Stop]
    max_shift_time: float
    weights: Weights
    optimization: Optional[OptimizationConfig] = None
    travel_mode: Literal["driving", "walking"] = "driving"

    @field_validator('max_shift_time')
    @classmethod
    def validate_max_shift(cls, v):
        if v <= 0:
            raise ValueError('max_shift_time must be > 0')
        return v

    @field_validator('stops')
    @classmethod
    def validate_stops(cls, v):
        if len(v) == 0:
            raise ValueError('at least one stop is required')
        
        stop_ids = [s.id for s in v]
        if len(stop_ids) != len(set(stop_ids)):
            raise ValueError('stop IDs must be unique')
        
        return v


class VisitResult(BaseModel):
    stop_id: int
    arrival: float
    start_service: float
    finish: float
    wait: float
    lateness: float


class RouteResponse(BaseModel):
    """Enhanced response with comprehensive optimization details."""
    # Route information
    initial_route_order: List[int]
    optimized_route_order: List[int]
    
    # Cost metrics (separated for transparency)
    initial_cost: float
    optimized_cost: float
    total_travel_time: float
    total_wait_time: float
    total_lateness: float
    priority_adjusted_lateness: float
    total_penalty: float
    distance_cost: float
    wait_cost: float
    lateness_cost: float
    shift_overrun: float
    shift_overrun_cost: float
    
    # Feasibility and timing
    finish_time: float
    feasible: bool
    
    # Optimization details
    improvement_percent: float
    improvement_found: bool
    iterations_used: int
    stopped_by: str  # "max_iterations", "no_improvement_limit", "time_limit", "completed"
    algorithm_used: str
    
    # Detailed visit information
    visits: List[VisitResult]


class ScenarioCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    route: RouteRequest


class ScenarioSummary(BaseModel):
    id: int
    name: str
    stop_count: int
    created_at: datetime
    updated_at: datetime


class ScenarioDetail(ScenarioSummary):
    route: RouteRequest


class OptimizationHistorySummary(BaseModel):
    id: int
    scenario_id: Optional[int] = None
    scenario_name: Optional[str] = None
    created_at: datetime
    algorithm_used: str
    optimized_cost: float
    improvement_percent: float
    feasible: bool
    stop_count: int


class OptimizationHistoryDetail(OptimizationHistorySummary):
    request: RouteRequest
    response: RouteResponse


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    email: str = Field(min_length=5, max_length=255)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    identity: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=1, max_length=128)


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime


class AuthResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    user: UserResponse


class GeocodeResult(BaseModel):
    lat: float
    lon: float
    display_name: str
    address: dict = Field(default_factory=dict)


class RoadWaypoint(BaseModel):
    lat: float
    lng: float


class RoadRouteRequest(BaseModel):
    waypoints: List[RoadWaypoint]
    mode: Literal["driving", "walking"] = "driving"
