"""CWE-613: sessions that outlive logout.

Logout used to clear only two keys, leaving the session ID alive. It now destroys the
session server-side, so a copied PHPSESSID can't be replayed afterwards. Sessions also
expire after a fixed lifetime (cwechecks.php), and login issues a fresh session ID.
"""
import requests

from conftest import BASE, JANE, get, http_login


def test_session_id_changes_on_login():
    s = requests.Session()
    s.verify = False
    s.get(f"{BASE}/login.php")
    before = s.cookies.get("PHPSESSID")
    s.post(f"{BASE}/login.php", data={"username": JANE[0], "password": JANE[1]})
    assert s.cookies.get("PHPSESSID") not in (None, before), "Session ID not regenerated (fixation)"


def test_session_cookie_flags():
    r = requests.get(f"{BASE}/login.php", verify=False)
    cookie = r.headers["Set-Cookie"].lower()
    for flag in ("httponly", "secure", "samesite=strict"):
        assert flag in cookie, flag


def test_replayed_session_after_logout_is_rejected():
    s = http_login(*JANE)
    stolen = s.cookies.get("PHPSESSID")
    assert get(s, "/vaults/vault_details.php?vault_id=3").status_code == 200

    get(s, "/logout.php")

    attacker = requests.Session()
    attacker.verify = False
    attacker.cookies.set("PHPSESSID", stolen)
    r = get(attacker, "/vaults/vault_details.php?vault_id=3")
    assert r.status_code == 302 and "/login.php" in r.headers["Location"]
