"""Shared fixtures for the HusKey Manager security regression suite.

Run against the Docker stack:  docker compose up -d  &&  pytest tests/
Browser tests use headless Chrome via Selenium (Selenium Manager fetches the driver). HTTP-level tests use requests.
"""
import os
import subprocess

import pytest
import requests
import urllib3
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

BASE = "https://localhost"

# Seed accounts from database/init.sql
JANE = ("janedoe", "Doe12345")          # Owner of HR Vault (3) only
ADMIN = ("admin", "Sup3rS3cr3t@dm1n")   # Site administrator



def compose(*args):
    """Run a docker compose command against the stack and return stdout."""
    return subprocess.run(
        ["docker", "compose", *args], capture_output=True, text=True, check=True
    ).stdout


@pytest.fixture(autouse=True)
def reset_lockouts():
    """Clear failed-login history so one test's failures can't lock out the next."""
    compose(
        "exec", "-T", "backend-mysql-database", "sh", "-c",
        'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" password_manager '
        '-e "DELETE FROM login_attempts" 2>/dev/null',
    )
    yield


@pytest.fixture
def browser():
    """Headless Chrome by default; set HUSKEY_BROWSER=firefox to use Firefox instead."""
    if os.environ.get("HUSKEY_BROWSER", "chrome").lower() == "firefox":
        options = webdriver.FirefoxOptions()
        options.add_argument("--headless")
        options.accept_insecure_certs = True
        driver = webdriver.Firefox(options=options)
    else:
        options = webdriver.ChromeOptions()
        options.add_argument("--headless=new")
        options.accept_insecure_certs = True  # self-signed local cert
        driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(30)
    driver.implicitly_wait(5)
    yield driver
    driver.quit()


def browser_login(driver, username, password):
    driver.get(f"{BASE}/login.php")
    driver.find_element(By.NAME, "username").send_keys(username)
    driver.find_element(By.NAME, "password").send_keys(password)
    form = driver.find_element(By.TAG_NAME, "form")
    driver.find_element(By.CSS_SELECTOR, "button[type=submit]").click()
    WebDriverWait(driver, 10).until(EC.staleness_of(form))  # wait for the server's response page


def http_login(username, password):
    """Return a requests.Session logged in as the given user (or not, if the login fails)."""
    s = requests.Session()
    s.verify = False
    s.post(f"{BASE}/login.php", data={"username": username, "password": password})
    return s


def get(session, path):
    """GET without following redirects, so tests can assert on 302s and 403s."""
    return session.get(f"{BASE}{path}", allow_redirects=False)
