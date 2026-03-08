"""Healthie EHR integration module.

This module provides functions to interact with Healthie for patient management
and appointment scheduling.
"""

import json
import os
import re
import time
from pathlib import Path

from loguru import logger
from playwright.async_api import Browser, BrowserContext, Page, async_playwright

_browser: Browser | None = None
_context: BrowserContext | None = None
_page: Page | None = None
_last_validated_at: float = 0.0  # epoch seconds of last successful session check

# Re-validate the session at most once every 5 minutes to avoid redundant navigations
_SESSION_REVALIDATE_INTERVAL = 300

# Cookie storage path — keeps Healthie from seeing a "new device" on every run
_COOKIE_FILE = Path(__file__).parent / ".healthie_session.json"

# Selector that is only present on authenticated pages (the sidebar nav)
_AUTH_SENTINEL = '[data-testid="main-nav"]'
# Selector for the global search bar on the home page
_SEARCH_SELECTOR = 'input[placeholder="Search Clients..."]'


async def login_to_healthie() -> Page:
    """Log into Healthie and return an authenticated page instance.

    Reuses an in-memory session if available and still valid, falling back to
    saved cookies, and finally to a full credential login. Re-validates the
    session at most once every 5 minutes to avoid redundant navigations.

    Returns:
        Page: An authenticated Playwright Page instance ready for use.

    Raises:
        ValueError: If required environment variables are missing.
        Exception: If login fails for any reason.
    """
    global _browser, _context, _page, _last_validated_at

    email = os.environ.get("HEALTHIE_EMAIL")
    password = os.environ.get("HEALTHIE_PASSWORD")

    if not email or not password:
        raise ValueError("HEALTHIE_EMAIL and HEALTHIE_PASSWORD must be set in environment variables")

    if _page is not None:
        # Skip re-validation if we validated recently — avoids redundant navigations
        # and race conditions when multiple calls come in close together.
        age = time.time() - _last_validated_at
        if age < _SESSION_REVALIDATE_INTERVAL:
            logger.info(f"Reusing existing Healthie session (validated {age:.0f}s ago)")
            return _page

        # Full session-validity check: navigate home and wait for the auth sentinel.
        try:
            await _page.goto("https://secure.gethealthie.com/", wait_until="domcontentloaded")
            # Wait up to 5s for either the authenticated sidebar or a redirect to login
            await _page.wait_for_selector(
                f'{_AUTH_SENTINEL}, [href*="sign_in"], [href*="account/login"]',
                timeout=5000,
            )
            current_url = _page.url
            if "sign_in" not in current_url and "account/login" not in current_url:
                logger.info(f"Reusing existing Healthie session (url={current_url})")
                _last_validated_at = time.time()
                return _page
            logger.warning(f"Existing session looks stale (url={current_url}), re-logging in")
        except Exception as e:
            logger.warning(f"Existing session check failed ({e}), re-logging in")
        # Reset so we start clean
        _page = None
        if _context:
            try:
                await _context.close()
            except Exception:
                pass
        _context = None
        if _browser:
            try:
                await _browser.close()
            except Exception:
                pass
        _browser = None

    logger.info("Logging into Healthie...")
    playwright = await async_playwright().start()
    _browser = await playwright.chromium.launch(headless=True)

    _context = await _browser.new_context(
        storage_state=str(_COOKIE_FILE) if _COOKIE_FILE.exists() else None,
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        viewport={"width": 1280, "height": 800},
        locale="en-US",
    )
    if _COOKIE_FILE.exists():
        logger.info(f"Loading saved session cookies from {_COOKIE_FILE}")
    _page = await _context.new_page()

    # Try saved cookies first — navigate and see if we land on an authenticated page
    await _page.goto("https://secure.gethealthie.com/", wait_until="domcontentloaded")
    try:
        await _page.wait_for_selector(
            f'{_AUTH_SENTINEL}, [href*="sign_in"], [href*="account/login"]',
            timeout=6000,
        )
    except Exception:
        pass

    current_url = _page.url
    if "sign_in" not in current_url and "account/login" not in current_url:
        logger.info(f"Saved session still valid, skipping login (url={current_url})")
        _last_validated_at = time.time()
        return _page

    # --- Full credential login ---
    logger.info("Saved session expired or invalid, logging in fresh...")
    await _page.goto("https://secure.gethealthie.com/users/sign_in", wait_until="domcontentloaded")

    email_input = _page.locator('input[type="email"], input[placeholder*="example"], input[name="email"]')
    await email_input.first.wait_for(state="visible", timeout=15000)
    await email_input.first.click()
    # Use type() instead of fill() so React's synthetic onChange fires
    await email_input.first.type(email, delay=40)
    logger.info(f"Typed email: {email}")

    log_in_btn = _page.locator('button:has-text("Log In"), button[type="submit"]')
    await log_in_btn.first.wait_for(state="visible", timeout=10000)
    await log_in_btn.first.click()
    logger.info("Clicked Log In button")

    # Wait for either the password field to appear or navigation away from the email step
    try:
        password_input = _page.locator('input[type="password"]')
        await password_input.wait_for(state="visible", timeout=8000)
        await password_input.click()
        await password_input.type(password, delay=40)
        logger.info("Typed password")
        submit_btn = _page.locator('button[type="submit"], button:has-text("Log In"), button:has-text("Sign In")')
        await submit_btn.first.click()
        logger.info("Clicked submit after password")
    except Exception as e:
        logger.warning(f"Password step: {e}")

    # Wait for the authenticated sentinel to appear (means login succeeded)
    await _page.wait_for_selector(_AUTH_SENTINEL, timeout=15000)

    current_url = _page.url
    logger.info(f"Post-login URL: {current_url}")
    if "sign_in" in current_url or "account/login" in current_url:
        raise Exception(f"Login may have failed - still on login page: {current_url}")

    await _context.storage_state(path=str(_COOKIE_FILE))
    logger.info(f"Session cookies saved to {_COOKIE_FILE}")
    logger.info("Successfully logged into Healthie")
    _last_validated_at = time.time()
    return _page


async def _reset_session() -> None:
    """Close the current browser/context and clear module-level session globals."""
    global _page, _context, _browser
    _page = None
    if _context:
        try:
            await _context.close()
        except Exception:
            pass
    _context = None
    if _browser:
        try:
            await _browser.close()
        except Exception:
            pass
    _browser = None


async def find_patient(name: str, date_of_birth: str) -> dict | None:
    """Find a patient in Healthie by name and date of birth."""
    try:
        page = await login_to_healthie()

        logger.info(f"Searching for patient: {name!r}")

        # Navigate home to ensure the Search Clients bar is mounted.
        # Use wait_for_selector on the search input itself rather than a fixed sleep.
        await page.goto("https://secure.gethealthie.com/", wait_until="domcontentloaded")
        await page.wait_for_selector(_SEARCH_SELECTOR, timeout=10000)

        logger.info(f"Current page URL before search: {page.url}")

        search_input = page.locator(_SEARCH_SELECTOR)

        async def _search_by_name(search_term: str) -> str | None:
            """Type into the search bar and return the first /users/<id> href found."""
            await search_input.click()
            await search_input.fill("")
            await search_input.fill(search_term)
            # Wait for a result link to appear rather than sleeping a fixed 2.5s
            try:
                await page.wait_for_selector('a[href*="/users/"]', timeout=4000)
            except Exception:
                pass
            for selector in ['a[href*="/users/"]', '[class*="search"] a', '[class*="dropdown"] a']:
                links = page.locator(selector)
                count = await links.count()
                for i in range(min(count, 10)):
                    href = await links.nth(i).get_attribute("href")
                    if href and re.search(r"/users/\d+", href):
                        return href
            return None

        # Try full name, then first name only (handles STT mis-splitting compound surnames)
        patient_url = await _search_by_name(name)
        if not patient_url:
            first_name = name.split()[0]
            logger.info(f"Full name {name!r} not found, retrying with first name {first_name!r}")
            patient_url = await _search_by_name(first_name)

        if not patient_url:
            logger.warning(f"No patient found for name: {name!r}")
            return None

        id_match = re.search(r"/users/(\d+)", patient_url)
        if not id_match:
            logger.warning(f"Could not extract patient ID from URL: {patient_url}")
            return None

        patient_id = id_match.group(1)
        logger.info(f"Found patient ID: {patient_id}")

        # Navigate to profile to verify DOB — wait for the page to settle via sentinel
        await page.goto(f"https://secure.gethealthie.com/users/{patient_id}", wait_until="domcontentloaded")
        await page.wait_for_selector(_AUTH_SENTINEL, timeout=8000)

        dob_on_profile = None
        try:
            dob_label = page.locator("text=Date of birth")
            if await dob_label.count() > 0:
                dob_value = await page.locator("text=Date of birth ~ *, text=Date of birth + *").first.text_content(timeout=2000)
                dob_on_profile = (dob_value or "").strip()
        except Exception:
            pass

        # Only reject if DOB is present on profile AND doesn't match what was provided
        if dob_on_profile and date_of_birth:
            digits_provided = re.sub(r"\D", "", date_of_birth)
            digits_on_profile = re.sub(r"\D", "", dob_on_profile)
            if digits_provided and digits_on_profile and digits_provided != digits_on_profile:
                logger.warning(f"DOB mismatch: provided={date_of_birth!r}, on profile={dob_on_profile!r}")
                return None

        return {
            "patient_id": patient_id,
            "name": name,
            "date_of_birth": dob_on_profile or date_of_birth,
        }

    except Exception as e:
        logger.error(f"Error finding patient '{name}': {e}")
        return None


async def create_appointment(patient_id: str, date: str, time: str) -> dict | None:
    """Create an appointment in Healthie for the specified patient."""
    try:
        page = await login_to_healthie()

        logger.info(f"Creating appointment for patient {patient_id} on {date} at {time}")

        await page.goto(f"https://secure.gethealthie.com/users/{patient_id}", wait_until="domcontentloaded")
        await page.wait_for_selector(_AUTH_SENTINEL, timeout=8000)

        if "sign_in" in page.url or "account/login" in page.url:
            logger.warning("Session expired, re-logging in for create_appointment...")
            await _reset_session()
            page = await login_to_healthie()
            await page.goto(f"https://secure.gethealthie.com/users/{patient_id}", wait_until="domcontentloaded")
            await page.wait_for_selector(_AUTH_SENTINEL, timeout=8000)

        # Format date as MM/DD/YYYY
        from datetime import datetime
        date_formatted = date
        for fmt in ("%B %d, %Y", "%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y"):
            try:
                parsed = datetime.strptime(date.strip(), fmt)
                date_formatted = parsed.strftime("%m/%d/%Y")
                break
            except ValueError:
                continue
        logger.info(f"Formatted date: {date_formatted}")

        normalised_time = _normalise_time(time)

        # Click "+ Add appointment" on the patient profile
        add_appt_btn = page.locator('a:has-text("Add appointment"), button:has-text("Add appointment")')
        await add_appt_btn.first.wait_for(state="visible", timeout=15000)
        await add_appt_btn.first.click()

        # Wait for the modal to appear
        await page.wait_for_selector('#appointment_type_id', timeout=8000)

        # --- Appointment Type (React Select) ---
        # React Select opens on mousedown, not click
        appt_type_result = await page.evaluate("""
            async () => {
                const input = document.querySelector('#appointment_type_id');
                if (!input) return 'input not found';
                let control = input.closest('[class*="control"]');
                if (!control) return 'control not found';
                control.dispatchEvent(new MouseEvent('mousedown', {bubbles: true, cancelable: true}));
                await new Promise(r => setTimeout(r, 600));
                const options = Array.from(document.querySelectorAll('[id^="react-select-appointment_type_id-option"]'));
                if (!options.length) return 'no options found';
                const text = options[0].textContent.trim();
                options[0].click();
                return text;
            }
        """)
        logger.info(f"Selected appointment type: {appt_type_result}")

        # Wait for the date input to be ready after selecting appointment type
        date_input = page.locator('input#date, input[name="date"]')
        await date_input.first.wait_for(state="visible", timeout=5000)

        # --- Start Date (use JS nativeInputValueSetter to trigger React onChange) ---
        await page.evaluate(f"""
            () => {{
                const input = document.querySelector('input#date, input[name="date"]');
                if (!input) return;
                const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
                nativeInputValueSetter.call(input, '{date_formatted}');
                input.dispatchEvent(new Event('input', {{ bubbles: true }}));
                input.dispatchEvent(new Event('change', {{ bubbles: true }}));
            }}
        """)
        await page.keyboard.press("Escape")  # Dismiss any datepicker popup
        logger.info(f"Filled date: {date_formatted}")

        # --- Start Time ---
        time_input = page.locator('input#time, input[name="time"], input[aria-labelledby="time"]')
        await time_input.first.wait_for(state="visible", timeout=5000)
        await time_input.first.click()
        # Give the dropdown a moment to render its list items
        await page.wait_for_timeout(500)
        clicked = await page.evaluate(f"""
            () => {{
                const items = Array.from(document.querySelectorAll('ul li'));
                const match = items.find(li => li.textContent.trim() === '{normalised_time}');
                if (match) {{
                    match.scrollIntoView({{block: 'center'}});
                    match.click();
                    return true;
                }}
                return false;
            }}
        """)
        if clicked:
            logger.info(f"Selected time: {normalised_time}")
        else:
            logger.warning(f"Time option '{normalised_time}' not found in dropdown, pressing Tab")
            await page.keyboard.press("Tab")

        await page.screenshot(path="/tmp/before_submit.png")
        logger.info("Screenshot saved to /tmp/before_submit.png")

        # --- Submit ---
        # Scroll modal to bottom so the button is in the viewport, then click
        await page.evaluate("""
            () => {
                const scrollables = [
                    document.querySelector('[role="dialog"]'),
                    document.querySelector('[class*="modal"]'),
                    document.querySelector('[class*="Modal"]'),
                ];
                for (const el of scrollables) {
                    if (el) { el.scrollTop = el.scrollHeight; break; }
                }
            }
        """)

        submit_btn = page.locator('button:has-text("Add appointment")').last
        await submit_btn.wait_for(state="visible", timeout=5000)
        await submit_btn.scroll_into_view_if_needed()
        await submit_btn.click()
        logger.info("Clicked submit button")

        # Wait for the modal to close (appointment was saved) rather than sleeping
        try:
            await page.wait_for_selector('[role="dialog"]', state="hidden", timeout=8000)
        except Exception:
            pass

        await page.screenshot(path="/tmp/after_submit.png")
        logger.info("Screenshot saved to /tmp/after_submit.png")

        error_locator = page.locator("text=Can't be blank")
        if await error_locator.count() > 0:
            error_text = await error_locator.all_text_contents()
            logger.warning(f"Appointment creation form errors: {error_text}")
            return None

        current_url = page.url
        appt_id_match = re.search(r"/appointments/(\d+)", current_url)
        appointment_id = appt_id_match.group(1) if appt_id_match else "created"

        logger.info(f"Appointment created via UI (id={appointment_id})")
        return {
            "appointment_id": appointment_id,
            "patient_id": patient_id,
            "date": date,
            "time": normalised_time,
        }

    except Exception as e:
        logger.error(f"Error creating appointment for patient {patient_id}: {e}")
        return None


def _normalise_time(time_str: str) -> str:
    """Normalise a time string to Healthie's expected format (e.g. '2:00 PM').

    Handles inputs like '14:00', '2pm', '2:00pm', '2:00 PM'.
    """
    time_str = time_str.strip()

    # Already in 12-hour format with AM/PM
    if re.match(r"^\d{1,2}:\d{2}\s*(AM|PM)$", time_str, re.IGNORECASE):
        parts = time_str.rsplit(" ", 1)
        return f"{parts[0]} {parts[1].upper()}"

    # "2pm" or "2am"
    m = re.match(r"^(\d{1,2})(am|pm)$", time_str, re.IGNORECASE)
    if m:
        hour = int(m.group(1))
        meridiem = m.group(2).upper()
        return f"{hour}:00 {meridiem}"

    # 24-hour format "14:00"
    m = re.match(r"^(\d{1,2}):(\d{2})$", time_str)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2))
        meridiem = "AM" if hour < 12 else "PM"
        display_hour = hour if hour <= 12 else hour - 12
        if display_hour == 0:
            display_hour = 12
        return f"{display_hour}:{minute:02d} {meridiem}"

    return time_str
