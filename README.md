# RouteMind

**RouteMind** is a route optimization API and web application built with FastAPI. It solves the Vehicle Routing Problem with Time Windows (VRPTW) by constructing an initial greedy route and then improving it using local-search algorithms. A built-in browser UI lets you interact with the API without any extra tooling.

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Optimization Algorithms](#optimization-algorithms)
- [Data Models](#data-models)
- [Rate Limiting](#rate-limiting)
- [Security](#security)
- [Running Tests](#running-tests)

---

## Features

- **User accounts** – register and log in; every action is scoped to the authenticated user.
- **JWT authentication** – stateless Bearer tokens (HS256, 7-day expiry).
- **Route optimization** – greedy construction followed by one of three local-search algorithms:
  - 2-opt (default)
  - Swap
  - Relocate
- **Time-window constraints** – each stop has an earliest and latest service window.
- **Priority scheduling** – stops are weighted by priority (1–5); late arrivals at high-priority stops are penalised more heavily.
- **Shift-time enforcement** – routes that exceed the maximum shift time are penalised.
- **Configurable cost weights** – tune the relative importance of distance, waiting time, lateness, priority, and shift overrun.
- **Scenario management** – save, retrieve, update, and delete named route scenarios.
- **Optimization history** – every run is persisted; browse past results at any time.
- **Rate limiting** – in-memory sliding-window limiter protects every write endpoint.
- **Web UI** – a static single-page application is served at `/`.
- **Health check** – `GET /health` returns the current version and status.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API framework | [FastAPI](https://fastapi.tiangolo.com/) |
| ASGI server | [Uvicorn](https://www.uvicorn.org/) |
| Data validation | [Pydantic v2](https://docs.pydantic.dev/) |
| Database | SQLite (via `sqlite3` stdlib) |
| Password hashing | `hashlib.scrypt` |
| JWT | Custom HS256 implementation (`hmac` + `hashlib`) |
| Frontend | Vanilla HTML/CSS/JS (served as static files) |
| Testing | `unittest` |

---

## Project Structure

```
RouteMind-Project/
└── routemind/
    ├── app/
    │   ├── main.py          # FastAPI application, routes, middleware
    │   ├── models.py        # Pydantic request/response models
    │   ├── auth.py          # User auth: registration, login, JWT, password hashing
    │   ├── storage.py       # Scenario and optimization-history persistence (SQLite)
    │   ├── optimizer.py     # Greedy construction + 2-opt / swap / relocate algorithms
    │   ├── evaluator.py     # Route cost evaluation (travel time, wait, lateness, penalties)
    │   ├── rate_limit.py    # In-memory sliding-window rate limiter
    │   ├── utils.py         # Distance helpers, distance-matrix builder
    ├── static/
    │   └── index.html       # Browser UI
    ├── tests/
    │   └── test_routemind.py
    ├── auth_store/          # SQLite auth database (auto-created)
    ├── data/                # App SQLite databases (auto-created)
    └── requirements.txt
```

---

## Getting Started

### Prerequisites

- Python 3.11+

### Install dependencies

```bash
cd routemind
pip install -r requirements.txt
```

### Run the server

```bash
uvicorn app.main:app --reload
```

The API is now available at `http://localhost:8000`.  
Open `http://localhost:8000/` in your browser to use the web UI.  
Interactive API docs are at `http://localhost:8000/docs`.

### Run with Docker

```bash
# From the repository root
docker build -t routemind .
docker run -p 8000:8000 routemind
```

Or with Docker Compose:

```bash
docker compose up
```

---

## Configuration

All configuration is done through environment variables:

| Variable | Description | Default |
|---|---|---|
| `ROUTEMIND_AUTH_DIR` | Directory for the auth SQLite database and secret-key file | `routemind/auth_store/` |
| `ROUTEMIND_DB_PATH` | Full path to the auth database file | `<AUTH_DIR>/routemind.db` |
| `ROUTEMIND_SECRET_KEY_PATH` | Full path to the JWT secret-key file | `<AUTH_DIR>/.secret_key` |
| `ROUTEMIND_SECRET_KEY` | JWT secret key value (overrides the file) | Auto-generated and persisted |

---

## API Reference

All endpoints that require authentication expect an `Authorization: Bearer <token>` header.

### Authentication

| Method | Path | Description | Auth required |
|---|---|---|---|
| `POST` | `/register` | Create a new account | No |
| `POST` | `/login` | Obtain a JWT | No |
| `GET` | `/me` | Get the current user's profile | Yes |

**Register / Login request body:**

```json
// POST /register
{ "username": "alice", "email": "alice@example.com", "password": "s3cr3tpassword" }

// POST /login  (identity = username or email)
{ "identity": "alice", "password": "s3cr3tpassword" }
```

Both return an `AuthResponse`:

```json
{
  "access_token": "<jwt>",
  "token_type": "bearer",
  "user": { "id": 1, "username": "alice", "email": "alice@example.com", "created_at": "..." }
}
```

---

### Route Optimization

| Method | Path | Description |
|---|---|---|
| `POST` | `/optimize` | Optimize a route and record the result |

Optional query parameter: `?scenario_id=<id>` links the run to a saved scenario.

**Request body:**

```json
{
  "depot": { "x": 0.0, "y": 0.0 },
  "stops": [
    {
      "id": 1,
      "x": 3.0, "y": 4.0,
      "window_start": 10.0, "window_end": 20.0,
      "service_time": 2.0,
      "priority": 3
    }
  ],
  "max_shift_time": 100.0,
  "weights": {
    "w_dist": 1.0,
    "w_wait": 1.0,
    "w_late": 2.0,
    "w_priority": 2.0,
    "w_shift": 4.0
  },
  "optimization": {
    "algorithm": "2opt",
    "max_iterations": 1000,
    "no_improvement_limit": 100,
    "time_limit": null
  }
}
```

**Response** includes the optimised route order, cost breakdown, feasibility flag, per-stop visit details, and metadata about the optimization run (algorithm used, iterations, improvement found, stopping reason).

---

### Scenarios

| Method | Path | Description |
|---|---|---|
| `GET` | `/scenarios` | List all saved scenarios |
| `POST` | `/scenarios` | Save or update a named scenario |
| `GET` | `/scenarios/{id}` | Get a scenario by ID |
| `DELETE` | `/scenarios/{id}` | Delete a scenario |

---

### Optimization History

| Method | Path | Description |
|---|---|---|
| `GET` | `/history` | List recent optimization runs (default: last 20, max 100) |
| `GET` | `/history/{run_id}` | Get the full request and response for a specific run |

---

### Utility

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Returns `{ "status": "operational", "version": "3.0" }` |
| `GET` | `/` | Serves the browser UI |

---

## Optimization Algorithms

### Greedy construction

Builds an initial route by repeatedly choosing the next unvisited stop with the lowest weighted cost (travel distance + expected wait + expected lateness), preferring feasible moves.

### 2-opt (default)

Iteratively reverses a segment of the route. Accepts the move if the new route has a lower total cost (or improves feasibility). Stops when no improvement is found for `no_improvement_limit` consecutive iterations, `max_iterations` is reached, or `time_limit` (seconds) expires.

### Swap

Considers every pair of stops and swaps them if that reduces total cost. Uses the same stopping criteria as 2-opt.

### Relocate

Considers moving each stop to every other position in the route. Accepts the move if it reduces total cost. Uses the same stopping criteria as 2-opt.

---

## Data Models

### Stop

| Field | Type | Description |
|---|---|---|
| `id` | int | Unique stop identifier |
| `x`, `y` | float | Coordinates |
| `window_start` | float | Earliest service time |
| `window_end` | float | Latest service time (must be ≥ `window_start`) |
| `service_time` | float | Time spent at the stop (must be ≥ 0) |
| `priority` | int (1–5) | 1 = lowest, 5 = highest |

### Weights

| Field | Default | Description |
|---|---|---|
| `w_dist` | 1.0 | Weight for total travel distance |
| `w_wait` | 1.0 | Weight for total waiting time |
| `w_late` | 2.0 | Weight for lateness |
| `w_priority` | 2.0 | Priority multiplier for lateness penalty |
| `w_shift` | 4.0 | Weight for shift-time overrun |

### Cost formula

```
total_cost = w_dist × travel_distance
           + w_wait × total_wait
           + w_late × priority_adjusted_lateness
           + w_shift × shift_overrun
```

where `priority_adjusted_lateness = Σ lateness_i × priority_factor(priority_i)` and
`priority_factor(p) = 1 + (p − 1) × 0.5 × max(w_priority, 0)`.

A stop with `priority=1` always has a factor of **1.0** (no extra penalty); each additional
priority level adds `0.5 × w_priority` to the multiplier, so a stop with `priority=5` and
`w_priority=2.0` carries a factor of **5.0** relative to a factor of 1.0 for `priority=1`.

---

## Rate Limiting

| Endpoint | Limit |
|---|---|
| `POST /register` | 5 requests / 60 s (per IP) |
| `POST /login` | 10 requests / 60 s (per IP) |
| `POST /optimize` | 30 requests / 60 s (per user) |
| `POST /scenarios` | 20 requests / 60 s (per user) |

Exceeding a limit returns HTTP `429 Too Many Requests` with a `Retry-After` header.

---

## Security

- **Password hashing** – scrypt (`N=16384, r=8, p=1`, 32-byte key, random 16-byte salt).
- **JWT** – HS256 signed with a 48-byte secret key; 7-day expiry. The key is auto-generated on first run and stored in `auth_store/.secret_key`, or it can be supplied via `ROUTEMIND_SECRET_KEY`.
- **CORS** – only `localhost` and `127.0.0.1` origins are allowed (with credentials).
- **Input validation** – all request bodies are validated by Pydantic before reaching business logic.

---

## Running Tests

```bash
cd routemind
python -m pytest tests/
# or
python -m unittest discover tests/
```