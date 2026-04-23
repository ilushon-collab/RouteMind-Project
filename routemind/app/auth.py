import base64
import binascii
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path


DEFAULT_AUTH_STORAGE_DIR = Path(__file__).resolve().parent.parent / "auth_store"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{3,32}$")
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_KEY_LEN = 32


def get_auth_storage_dir() -> Path:
    override = os.getenv("ROUTEMIND_AUTH_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_AUTH_STORAGE_DIR


def get_database_path() -> Path:
    override = os.getenv("ROUTEMIND_DB_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return get_auth_storage_dir() / "routemind.db"


def get_secret_key_path() -> Path:
    override = os.getenv("ROUTEMIND_SECRET_KEY_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return get_auth_storage_dir() / ".secret_key"


def reset_auth_caches() -> None:
    get_secret_key.cache_clear()


def init_auth_storage() -> None:
    get_auth_storage_dir().mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(get_database_path()) as connection:
        _configure_connection(connection)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.commit()


def get_connection() -> sqlite3.Connection:
    get_auth_storage_dir().mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(get_database_path())
    connection.row_factory = sqlite3.Row
    _configure_connection(connection)
    return connection


def validate_registration_input(username: str, email: str, password: str) -> tuple[str, str]:
    normalized_username = username.strip()
    normalized_email = normalize_email(email)

    if not USERNAME_PATTERN.fullmatch(normalized_username):
        raise ValueError("Username must be 3-32 characters and contain only letters, numbers, underscores, or hyphens.")

    if not EMAIL_PATTERN.fullmatch(normalized_email):
        raise ValueError("Enter a valid email address.")

    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")

    return normalized_username, normalized_email


def normalize_email(email: str) -> str:
    return email.strip().lower()


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    password_hash = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        dklen=SCRYPT_KEY_LEN,
    )
    return "$".join(
        [
            "scrypt",
            str(SCRYPT_N),
            str(SCRYPT_R),
            str(SCRYPT_P),
            _b64encode(salt),
            _b64encode(password_hash),
        ]
    )


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, n, r, p, salt, expected_hash = stored_hash.split("$")
        if algorithm != "scrypt":
            return False

        candidate_hash = hashlib.scrypt(
            password.encode("utf-8"),
            salt=_b64decode(salt),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=SCRYPT_KEY_LEN,
        )
        expected_hash_bytes = _b64decode(expected_hash)
    except (ValueError, TypeError, binascii.Error):
        return False

    return hmac.compare_digest(candidate_hash, expected_hash_bytes)


def create_user(username: str, email: str, password: str) -> dict:
    normalized_username, normalized_email = validate_registration_input(username, email, password)
    created_at = datetime.now(UTC).isoformat()
    password_hash = hash_password(password)

    with get_connection() as connection:
        existing = connection.execute(
            "SELECT id, username, email FROM users WHERE email = ? OR username = ?",
            (normalized_email, normalized_username),
        ).fetchall()

        for row in existing:
            if row["email"] == normalized_email:
                raise ValueError("An account with that email already exists.")
            if row["username"] == normalized_username:
                raise ValueError("That username is already taken.")

        cursor = connection.execute(
            """
            INSERT INTO users (username, email, password_hash, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (normalized_username, normalized_email, password_hash, created_at),
        )
        connection.commit()

        return {
            "id": cursor.lastrowid,
            "username": normalized_username,
            "email": normalized_email,
            "created_at": created_at,
        }


def authenticate_user(identity: str, password: str) -> dict | None:
    user_record = get_user_record_by_identity(identity)
    if not user_record or not verify_password(password, user_record["password_hash"]):
        return None
    return public_user_from_record(user_record)


def get_user_by_id(user_id: int) -> dict | None:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT id, username, email, password_hash, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    if not row:
        return None
    return public_user_from_record(row)


def get_user_record_by_identity(identity: str) -> sqlite3.Row | None:
    normalized_identity = identity.strip()
    if not normalized_identity:
        return None

    query = "SELECT id, username, email, password_hash, created_at FROM users WHERE lower(email) = lower(?) OR username = ?"
    with get_connection() as connection:
        return connection.execute(query, (normalized_identity, normalized_identity)).fetchone()


def create_access_token(user_id: int, expires_delta: timedelta | None = None) -> str:
    now = datetime.now(UTC)
    expiry = now + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    payload = {
        "sub": str(user_id),
        "exp": int(expiry.timestamp()),
        "iat": int(now.timestamp()),
    }
    return encode_jwt(payload)


def decode_access_token(token: str) -> dict:
    try:
        header_segment, payload_segment, signature_segment = token.split(".")
        provided_signature = _b64decode(signature_segment)
        header = json.loads(_b64decode_to_text(header_segment))
        payload = json.loads(_b64decode_to_text(payload_segment))
    except (ValueError, TypeError, json.JSONDecodeError, binascii.Error) as exc:
        raise ValueError("Invalid token format.") from exc

    signing_input = f"{header_segment}.{payload_segment}".encode("utf-8")
    expected_signature = hmac.new(
        get_secret_key().encode("utf-8"),
        signing_input,
        hashlib.sha256,
    ).digest()

    if not hmac.compare_digest(expected_signature, provided_signature):
        raise ValueError("Invalid token signature.")

    if header.get("alg") != "HS256":
        raise ValueError("Unsupported token algorithm.")

    exp = payload.get("exp")
    sub = payload.get("sub")

    if not isinstance(exp, int) or datetime.now(UTC).timestamp() >= exp:
        raise ValueError("Token has expired.")
    if not sub:
        raise ValueError("Token subject is missing.")

    return payload


def encode_jwt(payload: dict) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    header_segment = _b64encode_json(header)
    payload_segment = _b64encode_json(payload)
    signing_input = f"{header_segment}.{payload_segment}".encode("utf-8")
    signature = hmac.new(
        get_secret_key().encode("utf-8"),
        signing_input,
        hashlib.sha256,
    ).digest()
    signature_segment = _b64encode(signature)
    return f"{header_segment}.{payload_segment}.{signature_segment}"


def public_user_from_record(record: sqlite3.Row | dict) -> dict:
    return {
        "id": int(record["id"]),
        "username": record["username"],
        "email": record["email"],
        "created_at": record["created_at"],
    }


@lru_cache(maxsize=1)
def get_secret_key() -> str:
    env_secret = os.getenv("ROUTEMIND_SECRET_KEY")
    if env_secret:
        return env_secret

    storage_dir = get_auth_storage_dir()
    storage_dir.mkdir(parents=True, exist_ok=True)
    secret_key_path = get_secret_key_path()

    if secret_key_path.exists():
        return secret_key_path.read_text(encoding="utf-8").strip()

    secret = secrets.token_urlsafe(48)
    secret_key_path.write_text(secret, encoding="utf-8")
    return secret


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("utf-8")


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _b64encode_json(value: dict) -> str:
    return _b64encode(json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8"))


def _b64decode_to_text(value: str) -> str:
    return _b64decode(value).decode("utf-8")


def _configure_connection(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=MEMORY")
    connection.execute("PRAGMA synchronous=NORMAL")
