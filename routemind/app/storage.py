import json
from datetime import UTC, datetime

from app.auth import get_connection
from app.models import (
    OptimizationHistoryDetail,
    RouteRequest,
    RouteResponse,
    ScenarioDetail,
    ScenarioSummary,
)


def init_app_storage() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS saved_scenarios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                stop_count INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(user_id, name),
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS optimization_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                scenario_id INTEGER,
                scenario_name TEXT,
                request_json TEXT NOT NULL,
                response_json TEXT NOT NULL,
                algorithm_used TEXT NOT NULL,
                optimized_cost REAL NOT NULL,
                improvement_percent REAL NOT NULL,
                feasible INTEGER NOT NULL,
                stop_count INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id),
                FOREIGN KEY(scenario_id) REFERENCES saved_scenarios(id)
            )
            """
        )
        connection.commit()


def save_scenario(user_id: int, name: str, route: RouteRequest) -> ScenarioDetail:
    timestamp = datetime.now(UTC).isoformat()
    payload_json = json.dumps(route.model_dump(mode="json"))
    stop_count = len(route.stops)
    normalized_name = name.strip()

    with get_connection() as connection:
        existing = connection.execute(
            "SELECT id, created_at FROM saved_scenarios WHERE user_id = ? AND name = ?",
            (user_id, normalized_name),
        ).fetchone()

        if existing:
            connection.execute(
                """
                UPDATE saved_scenarios
                SET payload_json = ?, stop_count = ?, updated_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (payload_json, stop_count, timestamp, existing["id"], user_id),
            )
            connection.commit()
            scenario_id = int(existing["id"])
        else:
            cursor = connection.execute(
                """
                INSERT INTO saved_scenarios (user_id, name, payload_json, stop_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, normalized_name, payload_json, stop_count, timestamp, timestamp),
            )
            connection.commit()
            assert cursor.lastrowid is not None
            scenario_id = int(cursor.lastrowid)

    result = get_scenario(user_id, scenario_id)
    assert result is not None
    return result


def list_scenarios(user_id: int) -> list[ScenarioSummary]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, name, stop_count, created_at, updated_at
            FROM saved_scenarios
            WHERE user_id = ?
            ORDER BY updated_at DESC, id DESC
            """,
            (user_id,),
        ).fetchall()

    return [
        ScenarioSummary(
            id=int(row["id"]),
            name=row["name"],
            stop_count=int(row["stop_count"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
        for row in rows
    ]


def get_scenario(user_id: int, scenario_id: int) -> ScenarioDetail | None:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT id, name, payload_json, stop_count, created_at, updated_at
            FROM saved_scenarios
            WHERE user_id = ? AND id = ?
            """,
            (user_id, scenario_id),
        ).fetchone()

    if not row:
        return None

    return ScenarioDetail(
        id=int(row["id"]),
        name=row["name"],
        stop_count=int(row["stop_count"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        route=RouteRequest.model_validate(json.loads(row["payload_json"])),
    )


def delete_scenario(user_id: int, scenario_id: int) -> bool:
    with get_connection() as connection:
        cursor = connection.execute(
            "DELETE FROM saved_scenarios WHERE user_id = ? AND id = ?",
            (user_id, scenario_id),
        )
        connection.commit()
    return cursor.rowcount > 0


def record_optimization_run(
    user_id: int,
    request_model: RouteRequest,
    response_model: RouteResponse,
    scenario_id: int | None = None,
) -> int:
    timestamp = datetime.now(UTC).isoformat()
    request_json = json.dumps(request_model.model_dump(mode="json"))
    response_json = json.dumps(response_model.model_dump(mode="json"))
    scenario_name = None

    if scenario_id is not None:
        scenario = get_scenario(user_id, scenario_id)
        if scenario:
            scenario_name = scenario.name
        else:
            scenario_id = None

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO optimization_runs (
                user_id, scenario_id, scenario_name, request_json, response_json,
                algorithm_used, optimized_cost, improvement_percent, feasible, stop_count, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                scenario_id,
                scenario_name,
                request_json,
                response_json,
                response_model.algorithm_used,
                response_model.optimized_cost,
                response_model.improvement_percent,
                1 if response_model.feasible else 0,
                len(request_model.stops),
                timestamp,
            ),
        )
        connection.commit()

    assert cursor.lastrowid is not None
    return int(cursor.lastrowid)


def list_optimization_runs(user_id: int, limit: int = 20) -> list[dict]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, scenario_id, scenario_name, created_at, algorithm_used,
                   optimized_cost, improvement_percent, feasible, stop_count
            FROM optimization_runs
            WHERE user_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()

    return [
        {
            "id": int(row["id"]),
            "scenario_id": int(row["scenario_id"]) if row["scenario_id"] is not None else None,
            "scenario_name": row["scenario_name"],
            "created_at": row["created_at"],
            "algorithm_used": row["algorithm_used"],
            "optimized_cost": row["optimized_cost"],
            "improvement_percent": row["improvement_percent"],
            "feasible": bool(row["feasible"]),
            "stop_count": int(row["stop_count"]),
        }
        for row in rows
    ]


def get_optimization_run(user_id: int, run_id: int) -> OptimizationHistoryDetail | None:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT id, scenario_id, scenario_name, created_at, algorithm_used,
                   optimized_cost, improvement_percent, feasible, stop_count,
                   request_json, response_json
            FROM optimization_runs
            WHERE user_id = ? AND id = ?
            """,
            (user_id, run_id),
        ).fetchone()

    if not row:
        return None

    return OptimizationHistoryDetail(
        id=int(row["id"]),
        scenario_id=int(row["scenario_id"]) if row["scenario_id"] is not None else None,
        scenario_name=row["scenario_name"],
        created_at=row["created_at"],
        algorithm_used=row["algorithm_used"],
        optimized_cost=row["optimized_cost"],
        improvement_percent=row["improvement_percent"],
        feasible=bool(row["feasible"]),
        stop_count=int(row["stop_count"]),
        request=RouteRequest.model_validate(json.loads(row["request_json"])),
        response=RouteResponse.model_validate(json.loads(row["response_json"])),
    )
