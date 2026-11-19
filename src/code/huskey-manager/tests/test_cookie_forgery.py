"""CWE-565 / CWE-639: trusting client-side cookies for identity and role.

Originally `authenticated=<username>` and `isSiteAdministrator=1` were plain cookies, so
editing them in DevTools switched user or granted admin. Identity and role now live in the
server-side PHP session.
"""
import requests

from conftest import BASE, JANE, browser_login, get, http_login


def test_forged_identity_cookie_without_login_is_rejected():
    s = requests.Session()
    s.verify = False
    s.cookies.set("authenticated", "admin")
    s.cookies.set("isSiteAdministrator", "1")

    r = get(s, "/index.php")
    assert r.status_code == 302 and r.headers["Location"].endswith("/login.php")


def test_forged_admin_cookie_does_not_grant_admin(browser):
    browser_login(browser, *JANE)
    assert "Welcome" in browser.page_source

    browser.add_cookie({"name": "isSiteAdministrator", "value": "1", "path": "/"})
    browser.add_cookie({"name": "authenticated", "value": "admin", "path": "/"})
    browser.get(f"{BASE}/index.php")

    nav = browser.page_source
    assert 'href="/users/"' not in nav, "Users link shown to non-admin with forged cookie"
    assert 'href="/admin/"' not in nav, "Admin link shown to non-admin with forged cookie"


def test_non_admin_cannot_open_admin_pages_directly():
    s = http_login(*JANE)
    s.cookies.set("isSiteAdministrator", "1")

    for path in ("/users/index.php", "/admin/index.php"):
        r = get(s, path)
        assert r.status_code == 302 and r.headers["Location"].endswith("/index.php"), path
