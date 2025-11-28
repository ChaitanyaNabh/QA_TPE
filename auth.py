import os
import json
import hashlib
import hmac
import secrets

USERS_FILE = os.getenv("USERS_FILE", "users.json")
# For improved security set PASSWORD_SALT env var in your environment for this app
_PASSWORD_SALT = os.getenv("PASSWORD_SALT", "qa_ai_default_salt_change_this")


def _hash_password(password: str, salt: str = _PASSWORD_SALT) -> str:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def load_users() -> dict:
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_users(users: dict):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def create_user(username: str, password: str):
    if not username or not password:
        raise ValueError("username and password required")
    users = load_users()
    if username in users:
        raise ValueError("user already exists")
    users[username] = {"pw_hash": _hash_password(password)}
    save_users(users)


def verify_user(username: str, password: str) -> bool:
    users = load_users()
    if username not in users:
        return False
    stored = users[username].get("pw_hash", "")
    provided = _hash_password(password)
    return hmac.compare_digest(stored, provided)


def set_password(username: str, password: str):
    users = load_users()
    users[username] = {"pw_hash": _hash_password(password)}
    save_users(users)


def list_users() -> list:
    return list(load_users().keys())
