---
title: "HusKey Manager: Breaking & Hardening a Password Manager"
description: "Attacked and hardened a PHP/MySQL password manager running in Docker. I exploited SQL injection, cookie forgery, IDOR, path traversal, XSS and a file-upload reverse shell, fixed each at its root cause, and proved every fix with 27 pytest + Selenium tests that re-run the attacks against the live stack."
coverHtml: /projects/huskey/cover.html
keywords: [PHP, MySQL, SQL, Python, pytest, Selenium, Docker, Nginx, Web Security, OWASP Top 10, Penetration Testing, Secure Coding, Access Control, Metasploit, OWASP ZAP]
minorTags: ["INFO 310", "UW iSchool", Cybersecurity, "SQL Injection", XSS, IDOR, "Path Traversal", "File Upload", "Reverse Shell", "Brute Force", bcrypt, "Session Management", "Prepared Statements", RBAC, CWE, "CIA Triad", Monolog, Loggly, "Chrome DevTools", "Regression Testing"]
order: 3
lastUpdated: 2026-11-18
previews:
  - { label: vault-access.php (authorization), href: huskey-manager/webapp/public/components/vault-access.php }
  - { label: login.php, href: huskey-manager/webapp/public/login.php }
  - { label: download_file.php, href: huskey-manager/webapp/public/vaults/download_file.php }
  - { label: test_vault_access.py, href: huskey-manager/tests/test_vault_access.py }
  - { label: test_uploads.py, href: huskey-manager/tests/test_uploads.py }
  - { label: README.md, href: huskey-manager/README.md }
---

A password manager is the worst place to have a security bug: one flaw exposes every credential a team owns. HusKey Manager is a small PHP/MySQL password manager (nginx, PHP-FPM and MySQL in Docker) where teams keep shared passwords in vaults with Owner, Editor and Viewer roles. It started as a deliberately vulnerable app in UW INFO 310. My job was to think like an attacker: break it, find out *why* each attack worked, fix the cause rather than the symptom, and then prove the fix holds.

**The approach.** For each weakness I worked through the same loop: exploit it by hand (Chrome DevTools, OWASP ZAP, Metasploit), trace it to the line of code that trusted something it shouldn't, fix it at that trust boundary, and write a test that performs the same attack and asserts that it fails.

### What I broke, and how I fixed it

**1. SQL injection on login.** Typing `admin' -- ` as the username logged in as the administrator with no password. The query was built by pasting the input into SQL, so the `--` commented out the password check. Escaping the input would only patch this one form. The real fix is to stop mixing code and data, so every query in the app now uses **prepared statements**, and numeric IDs are cast to integers before they get anywhere near SQL.

![SQL injection payload rejected at the login page](/projects/huskey/sqli-blocked.png)

**2. Forged identity cookies.** The app stored *who you are* and *whether you're an admin* in plain cookies. Editing `authenticated=janedoe` to `admin` in DevTools switched accounts. The lesson is that the client can never be the source of truth for identity, so both values moved into the **server-side session**. The browser now only holds an opaque, random session ID, flagged `HttpOnly`, `Secure` and `SameSite=Strict`. That ID is regenerated at login and destroyed at logout.

**3. Broken access control (IDOR).** Logged in as a user who owns only the HR vault, changing `?vault_id=3` to `?vault_id=2` in the URL opened the Executives vault. The same trick worked on hidden form fields to edit or delete other teams' passwords. The root cause was that authorization was attached to *pages*, not to *actions*. I wrote one authorization helper, `vault-access.php`, that every vault read and write goes through. It looks up the user's role on that specific vault, and checks that the password row actually belongs to the vault named in the request.

![Changing vault_id in the URL to another team's vault is refused](/projects/huskey/idor-blocked.png)

**4. Path traversal on downloads.** `download_file.php?file=` accepted a path from the browser, so `../login.php` downloaded the app's own source. Now the server only ever opens a file path it stored itself for a vault the user can read. The client just names which one.

**5. File upload → remote code execution.** Using Metasploit, I generated a PHP Meterpreter payload, uploaded it as a vault attachment, and requested it, which opened a reverse shell on the server. Closing this took three independent layers:

- an extension allow-list with a size cap
- random server-side filenames
- an nginx rule that refuses to serve the uploads directory at all, so even a file that slips through can never execute

**6. Stored and reflected XSS.** A vault named `<script>…</script>` ran JavaScript in the browser of everyone who opened the vault list. The same was true of anything typed into the search box. All output is now HTML-escaped where it's rendered.

**7. Brute force.** Login allowed unlimited guesses. A counter kept in the session doesn't help, because an attacker can simply drop the cookie and start fresh. So failed attempts are recorded in the **database, per account and per IP**. After 5 failures in 15 minutes the login locks, even for the correct password.

![Sixth attempt with the correct password, cookies cleared, still locked out](/projects/huskey/bruteforce-locked.png)

**8. Credentials at rest and in logs.** Passwords are stored as **bcrypt** hashes. A **Monolog/Loggly** audit trail records logins, lockouts, blocked downloads and unauthorized vault actions, with the user and IP but never the submitted password. That way an incident can be traced afterwards without the logs becoming a second leak.

**9. Credentials sent in plain text.** The app ran over plain HTTP, so anyone on the same network could read every login. A Wireshark capture showed the username and password in the clear. nginx now serves the app only over **HTTPS (TLS)**, and the same capture shows only encrypted bytes.

![Wireshark capture of a login: readable username and password over HTTP (left), encrypted bytes over HTTPS (right)](/projects/huskey/tls-before-after.png)

### Proving it

A fix I can't demonstrate is a guess. The test suite (**pytest + Selenium + requests**, 27 tests, about 20 seconds) runs against the live Docker stack, and every test *is* an attack:

- SQL injection payloads on the login form and the search box
- forged cookies
- edited vault IDs on reads and on writes
- `../` paths sent to the download endpoint
- a PHP shell sent as an upload
- stored XSS planted in a vault name
- a scripted brute force that clears its cookies between attempts

Some tests check the opposite as well: a user can still open, and download from, their *own* vault, so a fix that simply blocks everything would fail.

![All 27 regression tests passing](/projects/huskey/pytest-passing.png)

**Stack:** PHP, MySQL, nginx, PHP-FPM, Docker, Python, pytest, Selenium, requests, bcrypt, Monolog/Loggly, Metasploit, OWASP ZAP, Wireshark, Chrome DevTools
