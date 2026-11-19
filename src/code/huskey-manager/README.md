# HusKey Manager: Breaking & Hardening a Password Manager

A PHP / MySQL password manager (nginx + PHP-FPM + MySQL in Docker) that started as a deliberately
vulnerable teaching app in UW INFO 310. I attacked it, patched each flaw at its root cause, and
wrote a pytest + Selenium suite that re-runs every attack against the live stack.

## What was fixed

| Weakness | Attack | Fix |
|---|---|---|
| SQL injection (CWE-89) | `admin' -- ` on login; `' OR '1'='1` in search | Prepared statements everywhere; IDs cast to `int` |
| Client-trusted identity (CWE-565) | Edit `authenticated` / `isSiteAdministrator` cookies in DevTools | Identity and role live only in the server-side session |
| Broken access control / IDOR (CWE-639) | Change `?vault_id=` or hidden form fields to read or modify another team's vault | `components/vault-access.php`: role check (Viewer / Editor / Owner) on every vault read and write, rows scoped to their vault |
| Path traversal (CWE-22) | `download_file.php?file=../login.php` | Only paths stored against the user's vault can be opened |
| Unrestricted upload → RCE (CWE-434) | Upload a PHP reverse shell as an attachment, then request it | Extension allow-list, size cap, random filenames, nginx denies `/vaults/uploads/` |
| Stored / reflected XSS (CWE-79) | Vault named `<script>…</script>`; script in search box | All output HTML-escaped |
| Brute force (CWE-307) | Unlimited password guesses | DB-backed lockout per account and per IP (5 failures / 15 min), survives clearing cookies |
| Plaintext credentials (CWE-256) | Database dump reveals every password | bcrypt (`password_hash` / `password_verify`) |
| Session handling (CWE-613, CWE-384) | Replay a copied `PHPSESSID` after logout; session fixation | Session regenerated on login, destroyed on logout, absolute timeout, `HttpOnly; Secure; SameSite=Strict` |
| Missing audit trail | No record of attacks | Monolog logging of logins, lockouts, blocked downloads and unauthorized vault actions (to Loggly or container stderr) |

## Run it

```bash
cp .env.example .env            # set MYSQL_PASSWORD
mkdir certs && openssl req -x509 -newkey rsa:2048 -nodes -days 30 \
  -keyout certs/localhost.key -out certs/localhost.crt \
  -subj "/CN=localhost" -addext "subjectAltName=DNS:localhost"
docker compose up --build -d     # https://localhost
```

Seed accounts are in `database/init.sql` (stored as bcrypt hashes).

## Test it

```bash
python -m venv .venv && .venv/bin/pip install pytest selenium requests
.venv/bin/pytest tests -v -c tests/pytest.ini     # 27 tests, ~20 s
```

Each test performs a real attack against the running stack and asserts it is blocked. Browser tests use
headless Chrome (`HUSKEY_BROWSER=firefox` to switch).

## Credit

The base application (UI, schema and the original vulnerable code) is the INFO 310 teaching
template from the UW Information School, MIT-licensed. See `LICENSE`.
