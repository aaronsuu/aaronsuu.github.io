"""CWE-256 / CWE-532: plaintext passwords at rest and in logs.

Passwords are stored as bcrypt hashes, and login events log the username and IP but never
the submitted password.
"""
import uuid

from conftest import compose, http_login


def query(sql):
    return compose(
        "exec", "-T", "backend-mysql-database", "sh", "-c",
        f'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" password_manager -N -e "{sql}" 2>/dev/null',
    )


def test_passwords_are_stored_as_bcrypt_hashes():
    rows = query("SELECT password FROM users").split()
    assert rows, "no users found"
    assert all(p.startswith("$2y$") and len(p) == 60 for p in rows)


def test_submitted_password_never_appears_in_logs():
    canary = f"canary-{uuid.uuid4().hex}"
    http_login("janedoe", canary)

    logs = compose("logs", "--no-color", "backend-php-server")
    assert "Login failed" in logs, "expected the failed login to be logged"
    assert canary not in logs, "submitted password was written to the logs"
