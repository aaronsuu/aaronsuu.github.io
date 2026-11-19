"""CWE-639 (IDOR) and CWE-22: reaching other teams' vaults and files by editing the URL.

janedoe owns only the HR Vault (id 3). Changing ?vault_id=2 used to show the Executives
Vault, and download_file.php served any path passed in ?file=. Both now check
vault_permissions for the logged-in user, and downloads only serve files recorded in that vault.
"""
from conftest import BASE, JANE, get, http_login


def test_own_vault_is_readable():
    r = get(http_login(*JANE), "/vaults/vault_details.php?vault_id=3")
    assert r.status_code == 200 and "hr_password1" in r.text


def test_other_vault_via_url_is_denied():
    r = get(http_login(*JANE), "/vaults/vault_details.php?vault_id=2")
    assert "exec_password1" not in r.text
    assert "do not have permission" in r.text


def test_download_from_other_vault_is_forbidden():
    r = get(http_login(*JANE), "/vaults/download_file.php?vault_id=2&file=exec_notes.txt")
    assert r.status_code == 403


def test_download_path_traversal_is_forbidden():
    s = http_login(*JANE)
    for f in ("../login.php", "../../vendor/autoload.php", "/etc/passwd"):
        r = get(s, f"/vaults/download_file.php?vault_id=3&file={f}")
        assert r.status_code == 403, f


def test_cannot_write_to_another_vault_by_editing_form_fields():
    """janedoe holds no role on the Developers Vault (1): every write must be refused."""
    s = http_login(*JANE)
    attempts = [
        ("/vaults/vault_details.php", {"deletePasswordId": 3, "vaultId": 1}),
        ("/vaults/vault_details.php", {"addUsername": "x", "addWebsite": "x", "addPassword": "x", "vaultId": 1}),
        ("/vaults/index.php", {"editVaultName": "pwned", "editVaultId": 1}),
        ("/vaults/index.php", {"deleteVaultId": 1}),
        ("/vaults/vault_permissions.php", {"action": "addPermission", "vault_id": 1, "user_id": 4, "role_id": 1}),
    ]
    for path, data in attempts:
        r = s.post(f"{BASE}{path}", data=data, allow_redirects=False)
        assert r.status_code == 403, (path, data)

    # Developers Vault is untouched
    admin = http_login("admin", "Sup3rS3cr3t@dm1n")
    page = get(admin, "/vaults/vault_details.php?vault_id=1").text
    assert "Developers Vault" in page and "developer1" in page


def test_cannot_reach_another_vaults_row_through_own_vault():
    """Password 3 lives in vault 1; pairing it with janedoe's own vault 3 must not work."""
    s = http_login(*JANE)
    r = s.post(f"{BASE}/vaults/vault_details.php",
               data={"deletePasswordId": 3, "vaultId": 3}, allow_redirects=False)
    assert r.status_code == 403


def test_download_from_own_vault_works():
    r = get(http_login(*JANE), "/vaults/download_file.php?vault_id=3&file=hr_notes.pdf")
    assert r.status_code == 200 and r.content.startswith(b"%PDF")
