import math
from dataclasses import dataclass
from typing import Protocol

from app.models import Depot, Stop

EARTH_RADIUS_KM = 6371.0


class DistanceProvider(Protocol):
    def distance(self, a: Depot | Stop, b: Depot | Stop) -> float:
        ...


class EuclideanDistanceProvider:
    def distance(self, a: Depot | Stop, b: Depot | Stop) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)


class HaversineDistanceProvider:
    """Returns the great-circle distance in kilometres between two geo points.

    Falls back to the node's (x, y) pair treated as (lng, lat) when the
    dedicated lat/lng fields are absent, which preserves backward-compatibility
    with any data that was saved before the address-based input was introduced.
    """

    def distance(self, a: Depot | Stop, b: Depot | Stop) -> float:
        lat1 = a.lat if a.lat is not None else a.y
        lng1 = a.lng if a.lng is not None else a.x
        lat2 = b.lat if b.lat is not None else b.y
        lng2 = b.lng if b.lng is not None else b.x

        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lng2 - lng1)
        a_val = (
            math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        )
        return EARTH_RADIUS_KM * 2 * math.atan2(math.sqrt(a_val), math.sqrt(1 - a_val))


class TravelTimeProvider:
    """Converts Haversine km distance to estimated travel time in **minutes**.

    This makes the route evaluator produce time-based metrics that are
    consistent with the max_shift_time field (which is also in minutes).
    Average speeds are intentionally conservative to account for traffic and
    non-straight road geometry.
    """

    SPEED_KMH: dict[str, float] = {"driving": 50.0, "walking": 5.0}

    def __init__(self, mode: str = "driving") -> None:
        self._speed_kmh = self.SPEED_KMH.get(mode, self.SPEED_KMH["driving"])
        self._haversine = HaversineDistanceProvider()

    def distance(self, a: Depot | Stop, b: Depot | Stop) -> float:
        km = self._haversine.distance(a, b)
        return km / self._speed_kmh * 60.0  # → minutes


@dataclass(frozen=True)
class DistanceMatrix:
    values: dict[str, dict[str, float]]

    def between(self, a: Depot | Stop, b: Depot | Stop) -> float:
        return self.values[node_key(a)][node_key(b)]


def node_key(node: Depot | Stop) -> str:
    if isinstance(node, Stop):
        return f"stop:{node.id}"
    return "depot"


def _has_geo_coords(node: Depot | Stop) -> bool:
    """Return True when the node carries real geographic lat/lng coordinates."""
    return node.lat is not None and node.lng is not None


def build_distance_matrix(
    depot: Depot,
    stops: list[Stop],
    provider: DistanceProvider | None = None,
    travel_mode: str = "driving",
) -> DistanceMatrix:
    if provider is None:
        # Automatically choose the appropriate provider: use real-world travel
        # time (minutes) when geographic coordinates are present, otherwise fall
        # back to dimensionless Euclidean distance for legacy abstract scenarios.
        provider = (
            TravelTimeProvider(mode=travel_mode)
            if _has_geo_coords(depot)
            else EuclideanDistanceProvider()
        )
    nodes: list[Depot | Stop] = [depot, *stops]
    values: dict[str, dict[str, float]] = {}

    for source in nodes:
        source_key = node_key(source)
        values[source_key] = {}
        for target in nodes:
            values[source_key][node_key(target)] = provider.distance(source, target)

    return DistanceMatrix(values=values)


def distance(
    a: Depot | Stop,
    b: Depot | Stop,
    distance_matrix: DistanceMatrix | None = None,
    provider: DistanceProvider | None = None,
) -> float:
    if distance_matrix is not None:
        return distance_matrix.between(a, b)
    return (provider or EuclideanDistanceProvider()).distance(a, b)


def priority_factor(priority: int, priority_weight: float = 1.0) -> float:
    scaled_weight = max(priority_weight, 0.0)
    return 1.0 + (priority - 1) * 0.5 * scaled_weight
