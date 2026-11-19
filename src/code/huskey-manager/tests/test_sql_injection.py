"""CWE-89: SQL injection on the login form.

The original query concatenated user input into SQL, so `admin' -- ` logged in as admin
without a password. Login now uses a prepared statement plus a bcrypt password check.
"""
import pytest

from conftest import BASE, browser_login

PAYLOADS = [
    "' OR '1'='1",
    "admin' -- ",
    "admin' AND 1=1;-- ",
    "' OR 1=1 LIMIT 1;#",
]


@pytest.mark.parametrize("payload", PAYLOADS)
def test_sql_injection_login_is_rejected(browser, payload):
    browser_login(browser, payload, "irrelevant")

    assert browser.current_url.startswith(f"{BASE}/login.php"), "Injection redirected past login"
    assert "Welcome" not in browser.page_source
    assert "Invalid username or password" in browser.page_source
