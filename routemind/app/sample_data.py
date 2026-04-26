sample_request = {
    "depot": {"x": 0, "y": 0},
    "stops": [
        {"id": 1, "x": 2, "y": 3, "window_start": 2, "window_end": 10, "service_time": 1, "priority": 3},
        {"id": 2, "x": 5, "y": 2, "window_start": 0, "window_end": 8, "service_time": 1, "priority": 5},
        {"id": 3, "x": 6, "y": 6, "window_start": 5, "window_end": 15, "service_time": 1, "priority": 2},
        {"id": 4, "x": 8, "y": 3, "window_start": 3, "window_end": 12, "service_time": 1, "priority": 4}
    ],
    "max_shift_time": 25,
    "weights": {
        "w_dist": 1.0,
        "w_wait": 0.5,
        "w_late": 3.0,
        "w_priority": 2.0,
        "w_shift": 4.0
    }
}