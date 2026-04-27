import time

from app.evaluator import evaluate_route
from app.models import Depot, Stop, Weights
from app.utils import DistanceMatrix, build_distance_matrix, distance, priority_factor


def is_better_evaluation(candidate_eval: dict, current_eval: dict) -> bool:
    if candidate_eval["feasible"] != current_eval["feasible"]:
        return candidate_eval["feasible"]
    return candidate_eval["total_cost"] < current_eval["total_cost"]


def greedy_initial_route(
    depot: Depot,
    stops: list[Stop],
    max_shift_time: float,
    weights: Weights,
    distance_matrix: DistanceMatrix | None = None,
):
    distance_matrix = distance_matrix or build_distance_matrix(depot, stops)
    unvisited = stops[:]
    route: list[Stop] = []
    current = depot
    current_time = 0.0

    while unvisited:
        best_feasible_stop = None
        best_feasible_score = float("inf")
        best_fallback_stop = None
        best_fallback_score = float("inf")

        for stop in unvisited:
            travel = distance(current, stop, distance_matrix=distance_matrix)
            arrival = current_time + travel
            expected_wait = max(0.0, stop.window_start - arrival)
            expected_lateness = max(0.0, arrival - stop.window_end)
            weighted_lateness = expected_lateness * priority_factor(stop.priority, weights.w_priority)

            score = (
                weights.w_dist * travel
                + weights.w_wait * expected_wait
                + weights.w_late * weighted_lateness
            )

            temp_route = route + [stop]
            eval_result = evaluate_route(
                depot,
                temp_route,
                max_shift_time,
                weights,
                distance_matrix=distance_matrix,
            )
            candidate_score = eval_result["total_cost"]

            if eval_result["feasible"] and score < best_feasible_score:
                best_feasible_score = score
                best_feasible_stop = stop

            if candidate_score < best_fallback_score:
                best_fallback_score = candidate_score
                best_fallback_stop = stop

        chosen_stop = best_feasible_stop or best_fallback_stop
        if chosen_stop is None:
            break

        route.append(chosen_stop)
        eval_result = evaluate_route(
            depot,
            route,
            max_shift_time,
            weights,
            distance_matrix=distance_matrix,
        )
        current_time = eval_result["finish_time"] - distance(chosen_stop, depot, distance_matrix=distance_matrix)
        current = chosen_stop
        unvisited.remove(chosen_stop)

    return route


def swap_move(route: list[Stop], i: int, j: int) -> list[Stop]:
    new_route = route[:]
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route


def relocate_move(route: list[Stop], i: int, j: int) -> list[Stop]:
    new_route = route[:]
    stop = new_route.pop(i)
    new_route.insert(j, stop)
    return new_route


def two_opt_swap(route: list[Stop], i: int, j: int) -> list[Stop]:
    return route[:i] + route[i : j + 1][::-1] + route[j + 1 :]


def local_search_swap(
    depot: Depot,
    route: list[Stop],
    max_shift_time: float,
    weights: Weights,
    max_iterations: int = 1000,
    no_improvement_limit: int = 100,
    time_limit: float = None,
    distance_matrix: DistanceMatrix | None = None,
) -> tuple[list[Stop], dict, dict]:
    if len(route) < 2:
        return route, evaluate_route(depot, route, max_shift_time, weights, distance_matrix=distance_matrix), {
            "iterations_used": 0,
            "improvement_found": False,
            "stopped_by": "completed",
        }

    best_route = route[:]
    best_eval = evaluate_route(depot, best_route, max_shift_time, weights, distance_matrix=distance_matrix)

    start_time = time.time()
    iterations = 0
    no_improvement_count = 0
    improvement_found = False
    stopped_by = "completed"

    while iterations < max_iterations and no_improvement_count < no_improvement_limit:
        if time_limit is not None and (time.time() - start_time) > time_limit:
            stopped_by = "time_limit"
            break

        improved = False

        for i in range(len(best_route)):
            for j in range(i + 1, len(best_route)):
                candidate = swap_move(best_route, i, j)
                candidate_eval = evaluate_route(
                    depot,
                    candidate,
                    max_shift_time,
                    weights,
                    distance_matrix=distance_matrix,
                )

                if is_better_evaluation(candidate_eval, best_eval):
                    best_route = candidate
                    best_eval = candidate_eval
                    improved = True
                    improvement_found = True
                    no_improvement_count = 0
                    break

            if improved:
                break

        if not improved:
            no_improvement_count += 1

        iterations += 1

    if no_improvement_count >= no_improvement_limit:
        stopped_by = "no_improvement_limit"
    elif iterations >= max_iterations:
        stopped_by = "max_iterations"

    metadata = {
        "iterations_used": iterations,
        "improvement_found": improvement_found,
        "stopped_by": stopped_by,
    }

    return best_route, best_eval, metadata


def local_search_relocate(
    depot: Depot,
    route: list[Stop],
    max_shift_time: float,
    weights: Weights,
    max_iterations: int = 1000,
    no_improvement_limit: int = 100,
    time_limit: float = None,
    distance_matrix: DistanceMatrix | None = None,
) -> tuple[list[Stop], dict, dict]:
    if len(route) < 2:
        return route, evaluate_route(depot, route, max_shift_time, weights, distance_matrix=distance_matrix), {
            "iterations_used": 0,
            "improvement_found": False,
            "stopped_by": "completed",
        }

    best_route = route[:]
    best_eval = evaluate_route(depot, best_route, max_shift_time, weights, distance_matrix=distance_matrix)

    start_time = time.time()
    iterations = 0
    no_improvement_count = 0
    improvement_found = False
    stopped_by = "completed"

    while iterations < max_iterations and no_improvement_count < no_improvement_limit:
        if time_limit is not None and (time.time() - start_time) > time_limit:
            stopped_by = "time_limit"
            break

        improved = False

        for i in range(len(best_route)):
            for j in range(len(best_route)):
                if i == j or abs(i - j) <= 1:
                    continue

                candidate = relocate_move(best_route, i, j)
                candidate_eval = evaluate_route(
                    depot,
                    candidate,
                    max_shift_time,
                    weights,
                    distance_matrix=distance_matrix,
                )

                if is_better_evaluation(candidate_eval, best_eval):
                    best_route = candidate
                    best_eval = candidate_eval
                    improved = True
                    improvement_found = True
                    no_improvement_count = 0
                    break

            if improved:
                break

        if not improved:
            no_improvement_count += 1

        iterations += 1

    if no_improvement_count >= no_improvement_limit:
        stopped_by = "no_improvement_limit"
    elif iterations >= max_iterations:
        stopped_by = "max_iterations"

    metadata = {
        "iterations_used": iterations,
        "improvement_found": improvement_found,
        "stopped_by": stopped_by,
    }

    return best_route, best_eval, metadata


def improve_route_2opt(
    depot: Depot,
    route: list[Stop],
    max_shift_time: float,
    weights: Weights,
    max_iterations: int = 1000,
    no_improvement_limit: int = 100,
    time_limit: float = None,
    distance_matrix: DistanceMatrix | None = None,
) -> tuple[list[Stop], dict, dict]:
    if len(route) < 2:
        return route, evaluate_route(depot, route, max_shift_time, weights, distance_matrix=distance_matrix), {
            "iterations_used": 0,
            "improvement_found": False,
            "stopped_by": "completed",
        }

    best_route = route[:]
    best_eval = evaluate_route(depot, best_route, max_shift_time, weights, distance_matrix=distance_matrix)

    start_time = time.time()
    iterations = 0
    no_improvement_count = 0
    improvement_found = False
    stopped_by = "completed"

    while iterations < max_iterations and no_improvement_count < no_improvement_limit:
        if time_limit is not None and (time.time() - start_time) > time_limit:
            stopped_by = "time_limit"
            break

        improved = False

        for i in range(len(best_route) - 1):
            for j in range(i + 1, len(best_route)):
                candidate = two_opt_swap(best_route, i, j)
                candidate_eval = evaluate_route(
                    depot,
                    candidate,
                    max_shift_time,
                    weights,
                    distance_matrix=distance_matrix,
                )

                if is_better_evaluation(candidate_eval, best_eval):
                    best_route = candidate
                    best_eval = candidate_eval
                    improved = True
                    improvement_found = True
                    no_improvement_count = 0
                    break

            if improved:
                break

        if not improved:
            no_improvement_count += 1

        iterations += 1

    if no_improvement_count >= no_improvement_limit:
        stopped_by = "no_improvement_limit"
    elif iterations >= max_iterations:
        stopped_by = "max_iterations"

    metadata = {
        "iterations_used": iterations,
        "improvement_found": improvement_found,
        "stopped_by": stopped_by,
    }

    return best_route, best_eval, metadata
