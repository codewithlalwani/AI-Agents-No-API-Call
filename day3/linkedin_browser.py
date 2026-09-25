"""Run directly to sign in manually; the scraper reuses the saved session."""

import os
import shlex
import sys
from pathlib import Path
from urllib.parse import urlparse

AUTH_FILE = Path(__file__).resolve().parent / ".auth" / "linkedin.json"
LOGIN_COMMAND = shlex.join([sys.executable, str(Path(__file__).resolve())])
LOGIN_HELP = (
    f"Run `{LOGIN_COMMAND}` in a terminal. Sign in in the browser it opens, "
    "open LinkedIn Jobs in the same tab, then press Enter in the terminal "
    "to save the session."
)


def playwright_api():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Run `python -m pip install playwright` and then "
            "`python -m playwright install chromium`."
        ) from exc
    return sync_playwright()


def check_login(page):
    path = urlparse(page.url).path.lower()
    if any(part in path for part in ("/login", "/checkpoint", "/authwall", "/uas/")):
        raise RuntimeError(f"LinkedIn requires login or verification. {LOGIN_HELP}")
    if page.locator('input[type="password"]').first.is_visible():
        raise RuntimeError(f"LinkedIn is showing a login form. {LOGIN_HELP}")


def save_login():
    with playwright_api() as p:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

        browser = p.chromium.launch(headless=False)
        try:
            context = browser.new_context()
            page = context.new_page()
            try:
                page.goto(
                    "https://www.linkedin.com/login",
                    wait_until="domcontentloaded",
                    timeout=15000,
                )
            except PlaywrightTimeoutError:
                print(
                    "LinkedIn is taking longer to load. The browser will stay open. "
                    "If needed, open https://www.linkedin.com/login in its address bar."
                )
            while True:
                input(
                    "Sign in in the opened browser and complete any verification. "
                    "Open your LinkedIn Jobs page in that tab, then press Enter here: "
                )
                try:
                    if urlparse(page.url).hostname not in {"linkedin.com", "www.linkedin.com"}:
                        raise RuntimeError("Return to LinkedIn in the opened tab before saving.")
                    check_login(page)
                except RuntimeError:
                    print(
                        "Login is not complete in this tab. Keep this browser open, "
                        "finish signing in, and press Enter again when Jobs is visible."
                    )
                    continue
                break
            AUTH_FILE.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            descriptor = os.open(AUTH_FILE, os.O_WRONLY | os.O_CREAT, 0o600)
            os.close(descriptor)
            AUTH_FILE.chmod(0o600)
            context.storage_state(path=str(AUTH_FILE))
            print(f"Session saved to {AUTH_FILE}. You can now run the Streamlit scraper.")
        finally:
            browser.close()


def scrape_linkedin(url):
    if not AUTH_FILE.exists():
        raise RuntimeError(f"No saved LinkedIn session found at {AUTH_FILE}. {LOGIN_HELP}")
    with playwright_api() as p:
        browser = p.chromium.launch(headless=False)
        try:
            context = browser.new_context(storage_state=str(AUTH_FILE))
            page = context.new_page()
            response = page.goto(url, wait_until="domcontentloaded", timeout=60000)
            check_login(page)
            if response and response.status >= 400:
                raise RuntimeError(f"LinkedIn returned HTTP {response.status}.")
            main = page.locator('main, [role="main"]').first
            try:
                main.wait_for(state="visible", timeout=20000)
                if urlparse(url).path.startswith("/jobs"):
                    main.locator('a[href*="/jobs/view/"]').first.wait_for(
                        state="visible", timeout=20000
                    )
            except Exception as exc:
                check_login(page)
                raise RuntimeError(
                    "LinkedIn content did not appear. The page may have no jobs, "
                    "require verification, or have changed its layout. " + LOGIN_HELP
                ) from exc
            check_login(page)
            content = main.inner_text().strip()
            if not content:
                raise RuntimeError("LinkedIn returned no readable content.")
            return content
        finally:
            browser.close()


if __name__ == "__main__":
    try:
        save_login()
    except (Exception, KeyboardInterrupt) as exc:
        raise SystemExit(f"Login was not saved: {exc}")
