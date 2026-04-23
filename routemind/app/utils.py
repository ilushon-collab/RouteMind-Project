import math
from dataclasses import dataclass
from typing import Protocol

from app.models import Depot, Stop


class DistanceProvider(Protocol):
    def distance(self, a: Depot | Stop, b: Depot | Stop) -> float:
        ...


class EuclideanDistanceProvider:
    def distance(self, a: Depot | Stop, b: Depot | Stop) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)


@dataclass(frozen=True)
class DistanceMatrix:
    values: dict[str, dict[str, float]]

    def between(self, a: Depot | Stop, b: Depot | Stop) -> float:
        return self.values[node_key(a)][node_key(b)]


def node_key(node: Depot | Stop) -> str:
    if isinstance(node, Stop):
        return f"stop:{node.id}"
    return "depot"


def build_distance_matrix(
    depot: Depot,
    stops: list[Stop],
    provider: DistanceProvider | None = None,
) -> DistanceMatrix:
    provider = provider or EuclideanDistanceProvider()
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
