"""
Generic Regression Testing Framework using Selenium
====================================================
Supports: Unit, Integration, System, E2E, Smoke, Sanity, and Regression tests.

Architecture:
  - BaseTest: Common setup/teardown, logging, screenshots
  - TestCategory: Enum of test types (smoke/sanity/e2e/etc.)
  - PageObject: Base class for Page Object Model pattern
  - TestRunner: Discovers, filters, and executes tests
  - TestReporter: Aggregates results into HTML/JSON reports

Usage:
  python regression_framework.py --suite smoke --browser chrome --env staging
  python regression_framework.py --suite regression --parallel 4
  python regression_framework.py --tags login,checkout --headless
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
import unittest
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

from selenium import webdriver
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


# ============================================================================
# CONFIGURATION & ENUMS
# ============================================================================

class TestCategory(Enum):
    """Categories of tests. Each has its own semantics and typical scope."""
    UNIT = "unit"                    # Single function/method in isolation
    INTEGRATION = "integration"      # Multiple units working together
    SYSTEM = "system"                # Full system, internal boundaries
    E2E = "e2e"                      # End-to-end user journeys
    SMOKE = "smoke"                  # Build-verification: "does it even start?"
    SANITY = "sanity"                # Narrow, post-change: "did my fix work?"
    REGRESSION = "regression"        # Full re-run to catch reintroduced bugs


class TestStatus(Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"  # Framework/env error, distinct from assertion failure


class Severity(Enum):
    BLOCKER = "blocker"
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    TRIVIAL = "trivial"


@dataclass
class Config:
    """Global configuration. Overridable via CLI or env vars."""
    base_url: str = "https://example.com"
    browser: str = "chrome"
    headless: bool = True
    implicit_wait: int = 5
    explicit_wait: int = 15
    page_load_timeout: int = 30
    screenshot_on_failure: bool = True
    output_dir: Path = field(default_factory=lambda: Path("test_results"))
    environment: str = "staging"
    parallel_workers: int = 1
    retry_count: int = 1  # Flaky-test retry budget

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        return cls(
            base_url=args.base_url or os.getenv("TEST_BASE_URL", cls.base_url),
            browser=args.browser,
            headless=args.headless,
            environment=args.env,
            parallel_workers=args.parallel,
            output_dir=Path(args.output),
        )


# ============================================================================
# RESULT DATA STRUCTURES
# ============================================================================

@dataclass
class TestResult:
    name: str
    category: TestCategory
    status: TestStatus
    duration_sec: float
    severity: Severity = Severity.MAJOR
    tags: List[str] = field(default_factory=list)
    message: str = ""
    traceback: str = ""
    screenshot_path: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        d["status"] = self.status.value
        d["severity"] = self.severity.value
        return d


# ============================================================================
# DRIVER FACTORY
# ============================================================================

class DriverFactory:
    """Centralized WebDriver creation. Swap browsers without touching tests."""

    @staticmethod
    def create(config: Config) -> WebDriver:
        browser = config.browser.lower()

        if browser == "chrome":
            opts = ChromeOptions()
            if config.headless:
                opts.add_argument("--headless=new")
            opts.add_argument("--no-sandbox")
            opts.add_argument("--disable-dev-shm-usage")
            opts.add_argument("--window-size=1920,1080")
            opts.add_argument("--disable-gpu")
            driver = webdriver.Chrome(options=opts)

        elif browser == "firefox":
            opts = FirefoxOptions()
            if config.headless:
                opts.add_argument("--headless")
            driver = webdriver.Firefox(options=opts)

        else:
            raise ValueError(f"Unsupported browser: {browser}")

        driver.implicitly_wait(config.implicit_wait)
        driver.set_page_load_timeout(config.page_load_timeout)
        return driver


# ============================================================================
# PAGE OBJECT MODEL BASE
# ============================================================================

class BasePage(ABC):
    """
    Page Object Model base. Subclass per page to encapsulate selectors
    and interactions, keeping tests readable and selectors in one place.
    """

    def __init__(self, driver: WebDriver, config: Config):
        self.driver = driver
        self.config = config
        self.wait = WebDriverWait(driver, config.explicit_wait)

    @property
    @abstractmethod
    def url(self) -> str:
        """Relative path for this page (appended to base_url)."""
        ...

    def open(self) -> "BasePage":
        self.driver.get(f"{self.config.base_url}{self.url}")
        return self

    def find(self, locator: tuple) -> WebElement:
        return self.wait.until(EC.presence_of_element_located(locator))

    def click(self, locator: tuple) -> None:
        self.wait.until(EC.element_to_be_clickable(locator)).click()

    def type_text(self, locator: tuple, text: str, clear: bool = True) -> None:
        el = self.find(locator)
        if clear:
            el.clear()
        el.send_keys(text)

    def is_visible(self, locator: tuple, timeout: int = 3) -> bool:
        try:
            WebDriverWait(self.driver, timeout).until(
                EC.visibility_of_element_located(locator)
            )
            return True
        except TimeoutException:
            return False

    def get_text(self, locator: tuple) -> str:
        return self.find(locator).text


# ============================================================================
# TEST DECORATORS — classification & metadata
# ============================================================================

def test_meta(
    category: TestCategory,
    severity: Severity = Severity.MAJOR,
    tags: Optional[List[str]] = None,
):
    """Attach metadata to a test method. The runner reads these attributes."""
    def decorator(func: Callable) -> Callable:
        func._test_category = category
        func._test_severity = severity
        func._test_tags = tags or []
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        wrapper._test_category = category
        wrapper._test_severity = severity
        wrapper._test_tags = tags or []
        return wrapper
    return decorator


def retry_on_failure(max_attempts: int = 2, delay: float = 1.0):
    """Retry flaky tests. Useful for network-sensitive E2E tests."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except (AssertionError, WebDriverException) as e:
                    last_exc = e
                    if attempt < max_attempts:
                        logging.warning(
                            "Attempt %d/%d failed for %s: %s — retrying",
                            attempt, max_attempts, func.__name__, e,
                        )
                        time.sleep(delay)
            raise last_exc
        return wrapper
    return decorator


# ============================================================================
# BASE TEST CLASS
# ============================================================================

class BaseTest(unittest.TestCase):
    """
    All tests inherit from this. Handles driver lifecycle, logging,
    and failure screenshots. Subclasses implement test_* methods.
    """
    config: Config = Config()
    driver: Optional[WebDriver] = None
    _needs_driver: bool = True  # Pure unit tests can set this to False

    @classmethod
    def setUpClass(cls) -> None:
        cls.logger = logging.getLogger(cls.__name__)
        cls.config.output_dir.mkdir(parents=True, exist_ok=True)
        (cls.config.output_dir / "screenshots").mkdir(exist_ok=True)

    def setUp(self) -> None:
        self._start_time = time.time()
        self.logger.info("→ Starting: %s", self._testMethodName)
        if self._needs_driver:
            self.driver = DriverFactory.create(self.config)

    def tearDown(self) -> None:
        elapsed = time.time() - self._start_time
        outcome = self._outcome
        failed = any(err for _, err in getattr(outcome, "errors", []) if err) or \
                 any(err for _, err in getattr(outcome, "failures", []) if err)

        if failed and self.driver and self.config.screenshot_on_failure:
            self._capture_screenshot()

        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass

        self.logger.info(
            "← Finished: %s in %.2fs (%s)",
            self._testMethodName, elapsed, "FAIL" if failed else "PASS",
        )

    def _capture_screenshot(self) -> Optional[str]:
        if not self.driver:
            return None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.__class__.__name__}_{self._testMethodName}_{timestamp}.png"
        path = self.config.output_dir / "screenshots" / filename
        try:
            self.driver.save_screenshot(str(path))
            self.logger.info("Screenshot saved: %s", path)
            return str(path)
        except WebDriverException as e:
            self.logger.error("Could not capture screenshot: %s", e)
            return None


# ============================================================================
# SAMPLE TESTS — one per category, demonstrating framework usage
# ============================================================================

class ExampleLoginPage(BasePage):
    """Sample page object demonstrating the POM pattern."""
    url = "/login"
    USERNAME = (By.ID, "username")
    PASSWORD = (By.ID, "password")
    SUBMIT = (By.CSS_SELECTOR, "button[type='submit']")
    ERROR_MSG = (By.CLASS_NAME, "error-message")
    WELCOME = (By.ID, "welcome-banner")

    def login(self, user: str, pw: str) -> None:
        self.type_text(self.USERNAME, user)
        self.type_text(self.PASSWORD, pw)
        self.click(self.SUBMIT)


class UnitTests(BaseTest):
    """Unit tests: isolated logic, no browser needed."""
    _needs_driver = False

    @test_meta(TestCategory.UNIT, Severity.MINOR, tags=["validation"])
    def test_email_validator_accepts_valid_format(self):
        def is_valid_email(e: str) -> bool:
            return "@" in e and "." in e.split("@")[-1]
        self.assertTrue(is_valid_email("user@example.com"))

    @test_meta(TestCategory.UNIT, Severity.MINOR, tags=["validation"])
    def test_email_validator_rejects_missing_at(self):
        def is_valid_email(e: str) -> bool:
            return "@" in e and "." in e.split("@")[-1]
        self.assertFalse(is_valid_email("userexample.com"))


class SmokeTests(BaseTest):
    """Smoke: the absolute basics. If these fail, stop the pipeline."""

    @test_meta(TestCategory.SMOKE, Severity.BLOCKER, tags=["critical-path"])
    def test_homepage_loads(self):
        self.driver.get(self.config.base_url)
        self.assertIn("", self.driver.title)  # Any title at all
        self.assertEqual(self.driver.execute_script("return document.readyState"), "complete")

    @test_meta(TestCategory.SMOKE, Severity.BLOCKER, tags=["critical-path"])
    def test_login_page_reachable(self):
        page = ExampleLoginPage(self.driver, self.config).open()
        self.assertTrue(page.is_visible(ExampleLoginPage.SUBMIT, timeout=5))


class SanityTests(BaseTest):
    """Sanity: narrow verification after a specific change."""

    @test_meta(TestCategory.SANITY, Severity.MAJOR, tags=["login"])
    def test_invalid_credentials_show_error(self):
        page = ExampleLoginPage(self.driver, self.config).open()
        page.login("bad_user", "bad_pass")
        self.assertTrue(page.is_visible(ExampleLoginPage.ERROR_MSG))


class IntegrationTests(BaseTest):
    """Integration: two or more components interacting."""

    @test_meta(TestCategory.INTEGRATION, Severity.MAJOR, tags=["auth", "session"])
    @retry_on_failure(max_attempts=2)
    def test_login_creates_authenticated_session(self):
        page = ExampleLoginPage(self.driver, self.config).open()
        page.login("valid_user", "valid_pass")
        self.assertTrue(page.is_visible(ExampleLoginPage.WELCOME))
        cookies = {c["name"] for c in self.driver.get_cookies()}
        self.assertIn("session_id", cookies)


class SystemTests(BaseTest):
    """System: the whole app behind the UI, all internal services."""

    @test_meta(TestCategory.SYSTEM, Severity.CRITICAL, tags=["search"])
    def test_search_returns_results_from_backend(self):
        self.driver.get(f"{self.config.base_url}/search?q=laptop")
        results = self.driver.find_elements(By.CSS_SELECTOR, ".search-result")
        self.assertGreater(len(results), 0)


class E2ETests(BaseTest):
    """E2E: full user journey, production-like environment."""

    @test_meta(TestCategory.E2E, Severity.CRITICAL, tags=["checkout", "critical-path"])
    @retry_on_failure(max_attempts=2, delay=2.0)
    def test_complete_purchase_flow(self):
        # Login → browse → add to cart → checkout → confirm
        login = ExampleLoginPage(self.driver, self.config).open()
        login.login("valid_user", "valid_pass")

        self.driver.get(f"{self.config.base_url}/products/42")
        self.driver.find_element(By.ID, "add-to-cart").click()

        self.driver.get(f"{self.config.base_url}/checkout")
        self.driver.find_element(By.ID, "place-order").click()

        confirmation = WebDriverWait(self.driver, 20).until(
            EC.visibility_of_element_located((By.ID, "order-confirmation"))
        )
        self.assertIn("Order", confirmation.text)


class RegressionTests(BaseTest):
    """
    Regression: the long-running full suite. In practice this is a
    superset — the runner includes tests from other categories when
    --suite regression is passed. This class holds specific
    previously-broken-now-fixed scenarios.
    """

    @test_meta(TestCategory.REGRESSION, Severity.MAJOR, tags=["bug-1234"])
    def test_bug_1234_special_chars_in_username_no_longer_crash(self):
        page = ExampleLoginPage(self.driver, self.config).open()
        page.login("user+tag@ex.com", "p@ss!#$")
        # Should render an error, not crash
        self.assertTrue(
            page.is_visible(ExampleLoginPage.ERROR_MSG)
            or page.is_visible(ExampleLoginPage.WELCOME)
        )


# ============================================================================
# TEST RUNNER
# ============================================================================

class TestRunner:
    """
    Discovers tests, filters by category/tag, runs (optionally in parallel),
    and collects results.
    """

    # Which test classes participate in each suite. `regression` is a superset.
    SUITE_MAP: Dict[str, List[Type[BaseTest]]] = {
        "unit": [UnitTests],
        "smoke": [SmokeTests],
        "sanity": [SanityTests],
        "integration": [IntegrationTests],
        "system": [SystemTests],
        "e2e": [E2ETests],
        "regression": [
            UnitTests, SmokeTests, SanityTests,
            IntegrationTests, SystemTests, E2ETests, RegressionTests,
        ],
    }

    def __init__(self, config: Config, suite: str, tags: Optional[List[str]] = None):
        self.config = config
        self.suite = suite
        self.tags = set(tags) if tags else None
        self.results: List[TestResult] = []
        self.logger = logging.getLogger("TestRunner")

    def _collect_tests(self) -> List[tuple]:
        """Return list of (TestClass, method_name) tuples to execute."""
        classes = self.SUITE_MAP.get(self.suite, [])
        collected = []
        for cls in classes:
            for attr in dir(cls):
                if not attr.startswith("test_"):
                    continue
                method = getattr(cls, attr)
                if self.tags:
                    method_tags = set(getattr(method, "_test_tags", []))
                    if not (self.tags & method_tags):
                        continue
                collected.append((cls, attr))
        return collected

    def _run_single(self, test_tuple: tuple) -> TestResult:
        cls, method_name = test_tuple
        cls.config = self.config
        method = getattr(cls, method_name)
        category = getattr(method, "_test_category", TestCategory.REGRESSION)
        severity = getattr(method, "_test_severity", Severity.MAJOR)
        tags = getattr(method, "_test_tags", [])

        start = time.time()
        suite = unittest.TestLoader().loadTestsFromName(method_name, cls)
        result = unittest.TestResult()
        suite.run(result)
        duration = time.time() - start

        full_name = f"{cls.__name__}.{method_name}"

        if result.wasSuccessful() and not result.skipped:
            return TestResult(
                name=full_name, category=category, status=TestStatus.PASSED,
                duration_sec=duration, severity=severity, tags=tags,
            )
        if result.skipped:
            return TestResult(
                name=full_name, category=category, status=TestStatus.SKIPPED,
                duration_sec=duration, severity=severity, tags=tags,
                message=result.skipped[0][1],
            )
        # Distinguish failure (assertion) from error (framework/env)
        if result.failures:
            _, tb = result.failures[0]
            return TestResult(
                name=full_name, category=category, status=TestStatus.FAILED,
                duration_sec=duration, severity=severity, tags=tags,
                message="Assertion failed", traceback=tb,
            )
        _, tb = result.errors[0]
        return TestResult(
            name=full_name, category=category, status=TestStatus.ERROR,
            duration_sec=duration, severity=severity, tags=tags,
            message="Unexpected error", traceback=tb,
        )

    def run(self) -> List[TestResult]:
        tests = self._collect_tests()
        self.logger.info("Collected %d tests for suite=%s", len(tests), self.suite)

        if self.config.parallel_workers > 1:
            with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as ex:
                futures = {ex.submit(self._run_single, t): t for t in tests}
                for fut in as_completed(futures):
                    try:
                        self.results.append(fut.result())
                    except Exception as e:
                        cls, name = futures[fut]
                        self.results.append(TestResult(
                            name=f"{cls.__name__}.{name}",
                            category=TestCategory.REGRESSION,
                            status=TestStatus.ERROR,
                            duration_sec=0.0,
                            message=str(e),
                            traceback=traceback.format_exc(),
                        ))
        else:
            for t in tests:
                self.results.append(self._run_single(t))

        return self.results


# ============================================================================
# REPORTER
# ============================================================================

class TestReporter:
    """Writes JSON + HTML reports and prints a console summary."""

    def __init__(self, results: List[TestResult], output_dir: Path):
        self.results = results
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def summary(self) -> Dict[str, int]:
        counts = {s.value: 0 for s in TestStatus}
        for r in self.results:
            counts[r.status.value] += 1
        counts["total"] = len(self.results)
        return counts

    def write_json(self) -> Path:
        path = self.output_dir / "results.json"
        payload = {
            "generated_at": datetime.utcnow().isoformat(),
            "summary": self.summary(),
            "tests": [r.to_dict() for r in self.results],
        }
        path.write_text(json.dumps(payload, indent=2))
        return path

    def write_html(self) -> Path:
        path = self.output_dir / "report.html"
        rows = []
        for r in self.results:
            color = {
                "PASSED": "#2ecc71", "FAILED": "#e74c3c",
                "ERROR": "#c0392b", "SKIPPED": "#95a5a6",
            }.get(r.status.value, "#333")
            rows.append(
                f"<tr><td>{r.name}</td><td>{r.category.value}</td>"
                f"<td style='color:{color};font-weight:bold'>{r.status.value}</td>"
                f"<td>{r.duration_sec:.2f}s</td><td>{r.severity.value}</td>"
                f"<td>{', '.join(r.tags)}</td></tr>"
            )
        summary = self.summary()
        html = f"""<!DOCTYPE html>
<html><head><title>Regression Report</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 2rem; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ padding: 8px 12px; border-bottom: 1px solid #eee; text-align: left; }}
th {{ background: #f4f4f4; }}
.summary {{ display: flex; gap: 1rem; margin: 1rem 0; }}
.card {{ padding: 1rem; border-radius: 8px; background: #f9f9f9; min-width: 100px; }}
</style></head><body>
<h1>Regression Test Report</h1>
<p>Generated: {datetime.utcnow().isoformat()}Z</p>
<div class="summary">
  <div class="card"><b>Total</b><br>{summary['total']}</div>
  <div class="card"><b>Passed</b><br>{summary['PASSED']}</div>
  <div class="card"><b>Failed</b><br>{summary['FAILED']}</div>
  <div class="card"><b>Errors</b><br>{summary['ERROR']}</div>
  <div class="card"><b>Skipped</b><br>{summary['SKIPPED']}</div>
</div>
<table>
<tr><th>Test</th><th>Category</th><th>Status</th><th>Duration</th><th>Severity</th><th>Tags</th></tr>
{''.join(rows)}
</table></body></html>"""
        path.write_text(html)
        return path

    def print_console(self) -> None:
        s = self.summary()
        print("\n" + "=" * 70)
        print(f"  RESULTS: {s['total']} tests | "
              f"✓ {s['PASSED']} | ✗ {s['FAILED']} | ! {s['ERROR']} | ⊘ {s['SKIPPED']}")
        print("=" * 70)
        for r in self.results:
            icon = {"PASSED": "✓", "FAILED": "✗", "ERROR": "!", "SKIPPED": "⊘"}[r.status.value]
            print(f"  {icon} [{r.category.value:11}] {r.name} ({r.duration_sec:.2f}s)")
            if r.status in (TestStatus.FAILED, TestStatus.ERROR) and r.message:
                print(f"      → {r.message}")
        print()


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generic Selenium Regression Framework")
    p.add_argument("--suite", default="smoke",
                   choices=list(TestRunner.SUITE_MAP.keys()),
                   help="Which test suite to run")
    p.add_argument("--browser", default="chrome", choices=["chrome", "firefox"])
    p.add_argument("--base-url", default=None, help="Target base URL")
    p.add_argument("--env", default="staging",
                   choices=["dev", "staging", "prod"])
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--no-headless", dest="headless", action="store_false")
    p.add_argument("--parallel", type=int, default=1, help="Parallel workers")
    p.add_argument("--tags", default="",
                   help="Comma-separated tags to filter (e.g. login,checkout)")
    p.add_argument("--output", default="test_results", help="Output directory")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    args = build_arg_parser().parse_args()
    setup_logging(args.verbose)
    config = Config.from_args(args)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    runner = TestRunner(config, args.suite, tags=tags or None)
    results = runner.run()

    reporter = TestReporter(results, config.output_dir)
    json_path = reporter.write_json()
    html_path = reporter.write_html()
    reporter.print_console()

    print(f"JSON report: {json_path}")
    print(f"HTML report: {html_path}")

    # Exit non-zero if any test failed — CI-friendly
    failed = any(r.status in (TestStatus.FAILED, TestStatus.ERROR) for r in results)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
