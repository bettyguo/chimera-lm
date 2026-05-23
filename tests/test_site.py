"""Tests for the MkDocs site build + accessibility self-check.

These tests are skipped automatically when mkdocs-material isn't installed
(local dev). They run in CI to catch broken links, missing pages, and a11y
regressions before the gh-pages deploy.
"""

from __future__ import annotations

import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SITE_DIR = REPO_ROOT / "site"


def _mkdocs_available() -> bool:
    try:
        import mkdocs  # noqa: F401
        import material  # noqa: F401  (mkdocs-material's package name)

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _mkdocs_available(),
    reason="mkdocs-material not installed (pip install mkdocs-material mkdocs-mermaid2-plugin)",
)


@pytest.fixture(scope="module")
def built_site() -> Path:
    """Run `mkdocs build --strict` once per module."""
    if SITE_DIR.exists():
        shutil.rmtree(SITE_DIR)
    result = subprocess.run(
        ["python", "-m", "mkdocs", "build", "--strict"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"mkdocs build failed:\n{result.stdout}\n{result.stderr}")
    assert SITE_DIR.exists(), "mkdocs build produced no site/ dir"
    return SITE_DIR


def test_site_builds_strict(built_site: Path) -> None:
    """The build itself succeeding (fixture) is the test."""
    assert (built_site / "index.html").exists()


@pytest.mark.parametrize(
    "rel",
    [
        "index.html",
        "architecture/index.html",
        "routing/index.html",
        "kv_cache/index.html",
        "think/index.html",
        "postmortem/index.html",
        "quickstart/index.html",
        "experiments/nano_report/index.html",
        "lit/index.html",
        "lit/01_gu_dao_2023_mamba/index.html",
        "lit/10_peng_2024_rwkv7/index.html",
        "decisions/ADR-001-cache-resolution/index.html",
        "decisions/ADR-002-toy-ssm-substitution/index.html",
        "sitemap.xml",
    ],
)
def test_required_pages_present(built_site: Path, rel: str) -> None:
    path = built_site / rel
    assert path.exists(), f"missing page: {rel}"
    assert path.stat().st_size > 0


class _A11yScan(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.headings: list[str] = []
        self.imgs_without_alt: list[str] = []
        self.empty_links: list[str] = []
        self.has_main = False
        self.has_skip_link = False
        self.has_lang = False
        self._link_open = False
        self._link_buf = ""
        self._link_attrs: dict = {}

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.headings.append(tag)
        if tag == "img" and not d.get("alt"):
            self.imgs_without_alt.append(d.get("src", "?"))
        if tag == "main":
            self.has_main = True
        if tag == "html" and d.get("lang"):
            self.has_lang = True
        if tag == "a":
            self._link_open = True
            self._link_attrs = d
            self._link_buf = ""
            if "md-skip" in d.get("class", ""):
                self.has_skip_link = True

    def handle_endtag(self, tag):
        if tag == "a" and self._link_open:
            text = self._link_buf.strip()
            if (
                not text
                and not self._link_attrs.get("aria-label")
                and not self._link_attrs.get("title")
            ):
                self.empty_links.append(self._link_attrs.get("href", "?"))
            self._link_open = False

    def handle_data(self, data):
        if self._link_open:
            self._link_buf += data


@pytest.mark.parametrize(
    "rel",
    [
        "index.html",
        "architecture/index.html",
        "routing/index.html",
        "kv_cache/index.html",
        "think/index.html",
        "postmortem/index.html",
        "experiments/nano_report/index.html",
        "lit/index.html",
        "decisions/ADR-001-cache-resolution/index.html",
    ],
)
def test_page_accessibility(built_site: Path, rel: str) -> None:
    """WCAG-aligned checks: heading hierarchy, alt text, link names, landmarks."""
    scan = _A11yScan()
    scan.feed((built_site / rel).read_text(encoding="utf-8"))

    issues: list[str] = []

    h1_count = sum(1 for h in scan.headings if h == "h1")
    if h1_count != 1:
        issues.append(f"{h1_count} <h1> tags (WCAG: exactly one per page)")

    last_lvl = 0
    for h in scan.headings:
        lvl = int(h[1])
        if last_lvl and lvl - last_lvl > 1:
            issues.append(f"heading level jump h{last_lvl} -> h{lvl}")
        last_lvl = lvl

    if scan.imgs_without_alt:
        issues.append(f"{len(scan.imgs_without_alt)} <img> without alt attribute")
    if scan.empty_links:
        issues.append(f"{len(scan.empty_links)} <a> without text/aria-label/title")
    if not scan.has_main:
        issues.append("no <main> landmark")
    if not scan.has_skip_link:
        issues.append("no skip-to-content link")
    if not scan.has_lang:
        issues.append("no <html lang> attribute")

    assert not issues, f"{rel} a11y issues:\n  - " + "\n  - ".join(issues)


# Pages we knowingly skip in the site-wide sweep, with the reason.
#
# - 404.html: Material's 404 page uses a stripped template that doesn't render
#   the `md-skip` skip-to-content link. We accept this because (a) the 404 is
#   never the primary content for any user journey, (b) it's framework-generated
#   so we can't fix it without a theme override, and (c) it still has the
#   <main> landmark, <html lang>, and proper headings.
_A11Y_SWEEP_SKIP = frozenset({"404.html"})


def test_all_pages_a11y_sweep(built_site: Path) -> None:
    """Defense-in-depth: scan every built HTML page for serious a11y faults.

    Same checks as the per-page test. Catches regressions on auto-generated
    pages we don't list above.
    """
    failures: list[str] = []
    for pg in built_site.rglob("*.html"):
        rel = str(pg.relative_to(built_site)).replace("\\", "/")
        if rel in _A11Y_SWEEP_SKIP:
            continue
        scan = _A11yScan()
        scan.feed(pg.read_text(encoding="utf-8"))
        if not scan.has_main:
            failures.append(f"{rel}: no <main>")
        if not scan.has_lang:
            failures.append(f"{rel}: no <html lang>")
        if not scan.has_skip_link:
            failures.append(f"{rel}: no skip-to-content link")
        if scan.empty_links:
            failures.append(f"{rel}: {len(scan.empty_links)} empty link(s)")
        if scan.imgs_without_alt:
            failures.append(f"{rel}: {len(scan.imgs_without_alt)} img(s) w/o alt")
        h1_count = sum(1 for h in scan.headings if h == "h1")
        if h1_count != 1:
            failures.append(f"{rel}: {h1_count} <h1> tags")
    assert not failures, "site-wide a11y sweep:\n  - " + "\n  - ".join(failures)
