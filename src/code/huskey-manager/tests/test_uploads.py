"""CWE-434: unrestricted file upload -> remote code execution.

The attachment form used to save any file, under its own name, inside the web root, so an
uploaded PHP payload (e.g. a Meterpreter reverse shell) ran when requested. Uploads are now
allow-listed, renamed, and the uploads directory is never served directly.
"""
from conftest import BASE, JANE, get, http_login

SHELL = b"<?php echo 'PWNED-' . (6 * 7); ?>"


def test_php_upload_is_rejected():
    s = http_login(*JANE)
    r = s.post(
        f"{BASE}/vaults/vault_details.php",
        data={"addUsername": "x", "addWebsite": "x", "addPassword": "x", "vaultId": 3},
        files={"file": ("shell.php", SHELL, "application/x-php")},
        allow_redirects=False,
    )
    assert r.status_code == 400 and "Upload rejected" in r.text
    assert "PWNED-42" not in get(s, "/vaults/uploads/shell.php").text


def test_double_extension_upload_is_rejected():
    s = http_login(*JANE)
    r = s.post(
        f"{BASE}/vaults/vault_details.php",
        data={"addUsername": "x", "addWebsite": "x", "addPassword": "x", "vaultId": 3},
        files={"file": ("notes.pdf.php", SHELL, "application/pdf")},
        allow_redirects=False,
    )
    assert r.status_code == 400


def test_uploads_directory_is_not_served_directly():
    s = http_login(*JANE)
    for f in ("hr_notes.pdf", "exec_notes.txt", "anything.php"):
        assert get(s, f"/vaults/uploads/{f}").status_code == 403, f
