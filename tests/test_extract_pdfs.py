"""Tests for the WHO PDF staging extractor.

Stays offline by mocking ``extract_full_text`` (the only function that
actually touches pdfplumber). Everything else -- regex, slug generation,
collision handling, body-length filter, front-matter capture -- is
pure-text logic that's easy to assert on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.midloop_pilot import extract_pdfs as ep


@pytest.fixture
def fake_pdf(monkeypatch):
    """Return a callable that installs canned (text, page_count) for
    a fake PDF path. Tests construct a Path that doesn't need to exist
    on disk; the monkeypatched extractor never reads from it."""
    state: dict = {}

    def install(text: str, n_pages: int = 1):
        state["text"] = text
        state["pages"] = n_pages

    def fake_extract(_path: Path) -> tuple[str, int]:
        return state["text"], state["pages"]

    monkeypatch.setattr(ep, "extract_full_text", fake_extract)
    return install


# ----------------------------------------------------------------- slugify

def test_slugify_basic() -> None:
    assert ep._slugify("Acute Abdomen") == "acute-abdomen"
    assert ep._slugify("  trim me  ") == "trim-me"
    assert ep._slugify("CO2 & H2O") == "co2-h2o"


def test_slugify_caps_at_60_chars() -> None:
    long = "x" * 200
    assert len(ep._slugify(long)) <= 60


def test_slugify_empty_falls_back_to_section() -> None:
    assert ep._slugify("...") == "section"
    assert ep._slugify("") == "section"


# ----------------------------------------------------------------- regex

def test_section_re_matches_numbered_heading() -> None:
    text = "1.2 Outline the adaptation process"
    m = ep._SECTION_RE.match(text)
    assert m is not None
    assert m.group("num") == "1.2"


def test_section_re_rejects_address_with_section_number_too_high() -> None:
    """1211 Geneva 27 Switzerland is an address, not a heading."""
    text = "1211 Geneva 27 Switzerland"
    assert ep._SECTION_RE.match(text) is None


def test_section_re_rejects_all_caps_banner() -> None:
    """Margin banners are ALL CAPS; require at least one lowercase
    letter in the title to filter them out."""
    text = "1 GUIDELINES FOR THE MANAGEMENT"
    assert ep._SECTION_RE.match(text) is None


def test_section_re_rejects_prose_starting_with_lowercase_word() -> None:
    """'5 children were enrolled' is prose, not a heading. The
    capital-letter requirement on the title's first character filters
    these out."""
    text = "5 children were enrolled in the trial"
    assert ep._SECTION_RE.match(text) is None


def test_section_re_accepts_prose_with_capitalized_first_word() -> None:
    """Documents the LIMITATION: '5 Patients are enrolled' DOES match.
    The body-length filter (>=250 chars) is the secondary defense -- a
    single prose sentence won't survive it. If a future PDF surfaces a
    false-positive that DOES exceed 250 chars, tighten the regex then."""
    text = "5 Patients are enrolled in the trial"
    assert ep._SECTION_RE.match(text) is not None


# ----------------------------------------------------------------- dump mode

def test_dump_writes_one_file_with_header(tmp_path, fake_pdf) -> None:
    fake_pdf("Body text of the PDF.\nSecond line.", n_pages=3)
    out = ep.dump_pdf(Path("/fake/snakebite.pdf"), tmp_path)
    assert out.name == "snakebite.md"
    content = out.read_text()
    assert "source_pdf: snakebite.pdf" in content
    assert "page_count: 3" in content
    assert "extraction_mode: dump" in content
    assert "Body text of the PDF." in content
    assert "promotion_status: STAGED" in content


# ----------------------------------------------------------------- sections mode

def test_sections_splits_on_numbered_headings(tmp_path, fake_pdf) -> None:
    body = (
        "Front matter prose before any section.\n"
        + ("paragraph " * 50) + "\n"
        + "1.1 Review current guidelines\n"
        + ("Body of section 1.1 with enough text to pass filter. " * 10) + "\n"
        + "1.2 Outline the adaptation process\n"
        + ("Body of section 1.2 with enough text to pass filter. " * 10) + "\n"
    )
    fake_pdf(body)
    written = ep.split_sections(Path("/fake/imci.pdf"), tmp_path)

    section_dir = tmp_path / "imci"
    assert section_dir.is_dir()
    names = sorted(p.name for p in written)
    assert "00-front-matter.md" in names
    assert any("1_1__" in n for n in names)
    assert any("1_2__" in n for n in names)


def test_sections_filters_short_bodies(tmp_path, fake_pdf) -> None:
    """Sections with <250 chars of body are dropped (TOC line
    items, address lines that slipped past the regex, etc.)."""
    body = (
        "1 Real section with a real body\n"
        + ("paragraph " * 50) + "\n"
        + "20 Avenue Appia heading-shaped address line\n"
        # Intentionally short body so the filter drops it.
    )
    fake_pdf(body)
    written = ep.split_sections(Path("/fake/x.pdf"), tmp_path)
    names = [p.name for p in written]
    assert any("1__real-section" in n for n in names)
    assert not any("20__avenue-appia" in n for n in names)


def test_sections_disambiguates_slug_collisions(tmp_path, fake_pdf) -> None:
    """A heading reprinted (TOC entry then real section) gets a
    __02 suffix instead of silently overwriting."""
    long = ("body " * 100)
    body = (
        "2.1 Make a preliminary list\n" + long + "\n"
        + "2.1 Make a preliminary list\n" + long + "\n"
    )
    fake_pdf(body)
    written = ep.split_sections(Path("/fake/x.pdf"), tmp_path)
    names = sorted(p.name for p in written)
    assert any(n.endswith("__02.md") for n in names), names


def test_sections_no_headings_falls_back_to_front_matter(tmp_path, fake_pdf) -> None:
    """Zero-heading PDF (e.g. postnatal care recommendations) writes
    everything to 00-front-matter.md so nothing is silently lost."""
    fake_pdf("Plain prose with no numbered headings whatsoever. " * 100)
    written = ep.split_sections(Path("/fake/postnatal.pdf"), tmp_path)
    assert len(written) == 1
    assert written[0].name == "00-front-matter.md"
    assert "Plain prose" in written[0].read_text()
