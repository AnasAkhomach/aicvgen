import sys
import os
import pytest
from jinja2 import Environment, FileSystemLoader, select_autoescape, Template
from pathlib import Path


def render_pdf_template(cv):
    template_dir = Path(__file__).parent.parent.parent / "src" / "templates"
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("pdf_template.html")
    return template.render(cv=cv)


def test_template_handles_missing_metadata():
    cv = type("CV", (), {"metadata": type("Meta", (), {})(), "sections": []})()
    html = render_pdf_template(cv)
    assert "Your Name" in html or "CV" in html


def test_template_handles_missing_fields_in_section():
    Section = type("Section", (), {})
    Item = type("Item", (), {})
    section = Section()
    section.name = "Professional Experience"
    section.items = [Item() for _ in range(2)]
    section.items[0].content = "Did something"
    section.items[1].content = None
    section.subsections = []
    cv = type("CV", (), {"metadata": type("Meta", (), {})(), "sections": [section]})()
    html = render_pdf_template(cv)
    assert "Did something" in html
    # Should not error on None content
    assert html.count("<li>") >= 1


def test_template_handles_subsection_metadata():
    Sub = type("Sub", (), {})
    sub = Sub()
    sub.name = "Project X"
    sub.metadata = type("Meta", (), {"company": None, "duration": None})()
    sub.items = []
    Section = type("Section", (), {})
    section = Section()
    section.name = "Professional Experience"
    section.items = []
    section.subsections = [sub]
    cv = type("CV", (), {"metadata": type("Meta", (), {})(), "sections": [section]})()
    html = render_pdf_template(cv)
    assert "Project X" in html
    # Should not error on missing company/duration
    assert "<h3>Project X</h3>" in html


def test_template_escapes_content():
    Section = type("Section", (), {})
    Item = type("Item", (), {})
    section = Section()
    section.name = "Key Qualifications"
    section.items = [Item() for _ in range(1)]
    section.items[0].content = "<script>alert('xss')</script>"
    section.subsections = []
    cv = type("CV", (), {"metadata": type("Meta", (), {})(), "sections": [section]})()
    html = render_pdf_template(cv)
    # The script tag should be escaped (Jinja2 escapes single quotes as &#39;)
    assert "&lt;script&gt;alert(&#39;xss&#39;)&lt;/script&gt;" in html
