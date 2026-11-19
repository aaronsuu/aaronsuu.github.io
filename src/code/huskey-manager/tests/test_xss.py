"""CWE-79: stored and reflected XSS.

A vault named `<script>...</script>` used to run in every viewer's browser, and the search
box reflected input straight into the page. All output is now HTML-escaped.
"""
import pytest

from conftest import BASE, JANE, compose, get, http_login

PAYLOAD = '<script>alert("xss")</script>'


def db(sql):
    return compose(
        "exec", "-T", "backend-mysql-database", "sh", "-c",
        f"mysql -uroot -p\"$MYSQL_ROOT_PASSWORD\" password_manager -N -e '{sql}' 2>/dev/null",
    ).strip()


@pytest.fixture
def xss_vault():
    s = http_login(*JANE)
    s.post(f"{BASE}/vaults/index.php", data={"vaultName": PAYLOAD})
    vault_id = db("SELECT MAX(vault_id) FROM vaults")
    yield s, vault_id
    s.post(f"{BASE}/vaults/index.php", data={"deleteVaultId": vault_id})


def test_stored_xss_in_vault_name_is_escaped(xss_vault):
    s, vault_id = xss_vault
    for path in ("/vaults/index.php", f"/vaults/vault_details.php?vault_id={vault_id}"):
        page = get(s, path).text
        assert PAYLOAD not in page, f"raw script tag rendered on {path}"
        assert "&lt;script&gt;" in page, f"vault name missing from {path}"


def test_reflected_xss_in_search_is_escaped():
    page = get(http_login(*JANE), f"/vaults/vault_details.php?vault_id=3&searchQuery={PAYLOAD}").text
    assert PAYLOAD not in page
    assert "&lt;script&gt;" in page


def test_sql_injection_in_search_returns_nothing_extra():
    s = http_login(*JANE)
    page = get(s, "/vaults/index.php?searchQuery=' OR '1'='1").text
    assert "Executives Vault" not in page and "Developers Vault" not in page
