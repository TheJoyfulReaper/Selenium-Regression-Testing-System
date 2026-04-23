# Selenium-Regression-Testing-System
Comprehensive Python regression testing framework using Selenium

Here's what the framework gives you:

Test categories — each has its own class and is tagged via the @test_meta decorator so the runner knows how to filter and report:

UnitTests — pure logic, no browser (opt out via _needs_driver = False)
SmokeTests — blocker-severity critical-path checks
SanityTests — narrow post-fix verification
IntegrationTests — multi-component interactions (login + session cookies)
SystemTests — full-app behind-the-UI
E2ETests — complete user journeys with retry support
RegressionTests — specific previously-broken scenarios; the regression suite also re-runs everything above as a superset

Core pieces

DriverFactory — browser setup (Chrome/Firefox, headless toggle) in one place
BasePage — Page Object Model base so tests stay readable and selectors live in one spot (ExampleLoginPage shows the pattern)
BaseTest — driver lifecycle, logging, auto-screenshots on failure
@test_meta + @retry_on_failure — classification metadata and flaky-test retry budget
TestRunner — collects by suite, filters by --tags, runs serial or parallel via ThreadPoolExecutor
TestReporter — JSON + styled HTML reports + console summary; CI-friendly non-zero exit on failure


A couple of notes on what to adapt before real use: the ExampleLoginPage selectors (#username, .error-message, etc.) are placeholders — point them at your actual app. The base_url defaults to example.com so smoke tests pass trivially; override with --base-url or the TEST_BASE_URL env var. And if you want Selenium 4's datetime.utcnow() deprecation warning silenced, swap those three calls to datetime.now(timezone.utc).
