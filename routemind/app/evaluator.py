from app.models import Depot, Stop, Weights
from app.utils import DistanceMatrix, distance, priority_factor


def evaluate_route(
    depot: Depot,
    route: list[Stop],
    max_shift_time: float,
    weights: Weights,
    distance_matrix: DistanceMatrix | None = None,
):
    current = depot
    current_time = 0.0

    visits = []
    total_travel = 0.0
    total_wait = 0.0
    total_lateness = 0.0
    priority_adjusted_lateness = 0.0

    for stop in route:
        travel = distance(current, stop, distance_matrix=distance_matrix)
        arrival = current_time + travel
        wait = max(0.0, stop.window_start - arrival)
        start_service = arrival + wait
        lateness = max(0.0, start_service - stop.window_end)
        finish = start_service + stop.service_time

        total_travel += travel
        total_wait += wait
        total_lateness += lateness
        priority_adjusted_lateness += lateness * priority_factor(stop.priority, weights.w_priority)

        visits.append(
            {
                "stop_id": stop.id,
                "arrival": arrival,
                "start_service": start_service,
                "finish": finish,
                "wait": wait,
                "lateness": lateness,
            }
        )

        current_time = finish
        current = stop

    return_travel = distance(current, depot, distance_matrix=distance_matrix)
    total_travel += return_travel
    finish_time = current_time + return_travel

    shift_overrun = max(0.0, finish_time - max_shift_time)
    feasible = shift_overrun == 0.0

    distance_cost = weights.w_dist * total_travel
    wait_cost = weights.w_wait * total_wait
    lateness_cost = weights.w_late * priority_adjusted_lateness
    shift_overrun_cost = weights.w_shift * shift_overrun

    total_penalty = wait_cost + lateness_cost + shift_overrun_cost
    total_cost = distance_cost + total_penalty

    return {
        "feasible": feasible,
        "total_travel_time": total_travel,
        "total_wait_time": total_wait,
        "total_lateness": total_lateness,
        "priority_adjusted_lateness": priority_adjusted_lateness,
        "total_penalty": total_penalty,
        "distance_cost": distance_cost,
        "wait_cost": wait_cost,
        "lateness_cost": lateness_cost,
        "shift_overrun": shift_overrun,
        "shift_overrun_cost": shift_overrun_cost,
        "total_cost": total_cost,
        "visits": visits,
        "finish_time": finish_time,
    }
