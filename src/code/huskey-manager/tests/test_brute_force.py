"""CWE-307: unlimited login attempts.

Failed logins are recorded in the database per account and per IP. After 5 failures in
15 minutes the login is locked, even with the correct password and even if the attacker
throws away their cookies between attempts.
"""
import requests

from conftest import BASE, JANE


def attempt(username, password):
    s = requests.Session()  # fresh session every time: no cookies carried over
    s.verify = False
    return s.post(f"{BASE}/login.php", data={"username": username, "password": password},
                  allow_redirects=False)


def test_lockout_after_five_failures_survives_cookie_reset():
    for _ in range(5):
        r = attempt(JANE[0], "wrong-password")
        assert "Invalid username or password" in r.text

    r = attempt(JANE[0], JANE[1])  # correct password, but locked
    assert r.status_code == 200
    assert "Too many failed attempts" in r.text


def test_successful_login_below_threshold():
    for _ in range(3):
        attempt(JANE[0], "wrong-password")
    r = attempt(JANE[0], JANE[1])
    assert r.status_code == 302 and r.headers["Location"].endswith("index.php")
