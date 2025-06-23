import pytest
from src.utils.prompt_utils import load_prompt_template, format_prompt
import tempfile
import os


def test_load_prompt_template_success():
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        f.write("Hello, {name}!")
        f.flush()
        path = f.name
    try:
        template = load_prompt_template(path)
        assert template == "Hello, {name}!"
    finally:
        os.remove(path)


def test_load_prompt_template_fallback():
    template = load_prompt_template("nonexistent.txt", fallback="Fallback {x}")
    assert template == "Fallback {x}"


def test_format_prompt_success():
    template = "Hello, {name}!"
    result = format_prompt(template, name="World")
    assert result == "Hello, World!"


def test_format_prompt_error():
    template = "Hello, {name}!"
    with pytest.raises(KeyError):
        format_prompt(template, foo="bar")
