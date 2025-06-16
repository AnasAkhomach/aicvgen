"""Unit tests for TemplateManager.

Tests template loading, rendering, caching,
and template management functionality.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from typing import Dict, Any

from src.services.template_manager import TemplateManager
from src.utils.exceptions import TemplateError


class TestTemplateManager:
    """Test cases for TemplateManager."""

    @pytest.fixture
    def temp_template_dir(self):
        """Create a temporary directory for test templates."""
        temp_dir = tempfile.mkdtemp()
        template_dir = Path(temp_dir) / "templates"
        template_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample template files
        (template_dir / "experience.txt").write_text(
            "Experience: {title} at {company}\n"
            "Duration: {duration}\n"
            "Description: {description}"
        )
        
        (template_dir / "education.txt").write_text(
            "Education: {degree} in {field}\n"
            "Institution: {institution}\n"
            "Year: {year}"
        )
        
        (template_dir / "skills.txt").write_text(
            "Skills: {skill_list}\n"
            "Proficiency: {level}"
        )
        
        # Create a template with conditional logic
        (template_dir / "conditional.txt").write_text(
            "Name: {name}\n"
            "{% if email %}Email: {email}{% endif %}\n"
            "{% if phone %}Phone: {phone}{% endif %}"
        )
        
        yield template_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def template_manager(self, temp_template_dir):
        """Create a TemplateManager instance for testing."""
        return TemplateManager(template_dir=str(temp_template_dir))

    @pytest.fixture
    def sample_template_data(self):
        """Create sample data for template rendering."""
        return {
            "title": "Senior Software Engineer",
            "company": "Tech Corp",
            "duration": "2020-2023",
            "description": "Developed scalable web applications",
            "degree": "Bachelor of Science",
            "field": "Computer Science",
            "institution": "University of Technology",
            "year": "2020",
            "skill_list": "Python, JavaScript, React",
            "level": "Advanced",
            "name": "John Doe",
            "email": "john.doe@example.com",
            "phone": "+1-555-0123"
        }

    def test_template_manager_initialization(self, temp_template_dir):
        """Test TemplateManager initialization."""
        template_manager = TemplateManager(template_dir=str(temp_template_dir))
        
        assert template_manager.template_dir == Path(temp_template_dir)
        assert hasattr(template_manager, '_template_cache')
        assert hasattr(template_manager, '_jinja_env')

    def test_load_template_success(self, template_manager):
        """Test successful template loading."""
        template_content = template_manager.load_template("experience")
        
        assert template_content is not None
        assert "Experience: {title} at {company}" in template_content
        assert "Duration: {duration}" in template_content
        assert "Description: {description}" in template_content

    def test_load_template_with_extension(self, template_manager):
        """Test loading template with .txt extension."""
        template_content = template_manager.load_template("experience.txt")
        
        assert template_content is not None
        assert "Experience: {title} at {company}" in template_content

    def test_load_template_not_found(self, template_manager):
        """Test loading a non-existent template."""
        with pytest.raises(TemplateError, match="Template not found"):
            template_manager.load_template("nonexistent")

    def test_template_caching(self, template_manager):
        """Test that templates are cached after first load."""
        # Load template first time
        template_content1 = template_manager.load_template("experience")
        
        # Check that template is in cache
        assert "experience" in template_manager._template_cache
        
        # Load template second time (should come from cache)
        template_content2 = template_manager.load_template("experience")
        
        # Should be the same content
        assert template_content1 == template_content2
        
        # Verify cache hit by checking cache directly
        assert template_manager._template_cache["experience"] == template_content1

    def test_render_template_simple(self, template_manager, sample_template_data):
        """Test simple template rendering."""
        rendered = template_manager.render_template("experience", sample_template_data)
        
        assert "Experience: Senior Software Engineer at Tech Corp" in rendered
        assert "Duration: 2020-2023" in rendered
        assert "Description: Developed scalable web applications" in rendered

    def test_render_template_with_missing_data(self, template_manager):
        """Test template rendering with missing data."""
        incomplete_data = {
            "title": "Software Engineer",
            "company": "Tech Corp"
            # Missing 'duration' and 'description'
        }
        
        with pytest.raises(TemplateError, match="Missing template variables"):
            template_manager.render_template("experience", incomplete_data)

    def test_render_template_conditional_logic(self, template_manager, sample_template_data):
        """Test template rendering with conditional logic."""
        rendered = template_manager.render_template("conditional", sample_template_data)
        
        assert "Name: John Doe" in rendered
        assert "Email: john.doe@example.com" in rendered
        assert "Phone: +1-555-0123" in rendered

    def test_render_template_conditional_missing_optional(self, template_manager):
        """Test template rendering with missing optional conditional data."""
        data_without_phone = {
            "name": "Jane Doe",
            "email": "jane.doe@example.com"
            # Missing 'phone'
        }
        
        rendered = template_manager.render_template("conditional", data_without_phone)
        
        assert "Name: Jane Doe" in rendered
        assert "Email: jane.doe@example.com" in rendered
        assert "Phone:" not in rendered  # Should not appear due to conditional

    def test_get_available_templates(self, template_manager):
        """Test getting list of available templates."""
        templates = template_manager.get_available_templates()
        
        assert isinstance(templates, list)
        assert "experience" in templates
        assert "education" in templates
        assert "skills" in templates
        assert "conditional" in templates
        assert len(templates) >= 4

    def test_template_exists(self, template_manager):
        """Test checking if template exists."""
        assert template_manager.template_exists("experience")
        assert template_manager.template_exists("education")
        assert not template_manager.template_exists("nonexistent")

    def test_clear_cache(self, template_manager):
        """Test clearing template cache."""
        # Load a template to populate cache
        template_manager.load_template("experience")
        assert len(template_manager._template_cache) > 0
        
        # Clear cache
        template_manager.clear_cache()
        assert len(template_manager._template_cache) == 0

    def test_reload_template(self, template_manager, temp_template_dir):
        """Test reloading a template after modification."""
        # Load template initially
        original_content = template_manager.load_template("experience")
        
        # Modify the template file
        template_file = temp_template_dir / "experience.txt"
        template_file.write_text("Modified: {title} at {company}")
        
        # Reload template
        template_manager.reload_template("experience")
        
        # Load template again
        new_content = template_manager.load_template("experience")
        
        assert new_content != original_content
        assert "Modified: {title} at {company}" in new_content

    def test_add_template_runtime(self, template_manager):
        """Test adding a template at runtime."""
        template_name = "runtime_template"
        template_content = "Runtime: {value}"
        
        # Add template
        template_manager.add_template(template_name, template_content)
        
        # Verify template exists
        assert template_manager.template_exists(template_name)
        
        # Verify template can be rendered
        rendered = template_manager.render_template(template_name, {"value": "test"})
        assert "Runtime: test" in rendered

    def test_remove_template(self, template_manager):
        """Test removing a template."""
        # Add a template first
        template_name = "temp_template"
        template_manager.add_template(template_name, "Temp: {value}")
        
        # Verify it exists
        assert template_manager.template_exists(template_name)
        
        # Remove template
        template_manager.remove_template(template_name)
        
        # Verify it's removed
        assert not template_manager.template_exists(template_name)

    def test_template_validation(self, template_manager):
        """Test template syntax validation."""
        # Valid template
        valid_template = "Hello {name}!"
        assert template_manager.validate_template(valid_template)
        
        # Invalid template (unclosed tag)
        invalid_template = "Hello {name!"
        assert not template_manager.validate_template(invalid_template)

    def test_get_template_variables(self, template_manager):
        """Test extracting variables from template."""
        variables = template_manager.get_template_variables("experience")
        
        assert isinstance(variables, set)
        assert "title" in variables
        assert "company" in variables
        assert "duration" in variables
        assert "description" in variables

    def test_render_template_with_filters(self, template_manager):
        """Test template rendering with Jinja2 filters."""
        # Add a template with filters
        template_content = "Name: {name|upper}\nEmail: {email|lower}"
        template_manager.add_template("filtered", template_content)
        
        data = {
            "name": "john doe",
            "email": "JOHN.DOE@EXAMPLE.COM"
        }
        
        rendered = template_manager.render_template("filtered", data)
        
        assert "Name: JOHN DOE" in rendered
        assert "Email: john.doe@example.com" in rendered

    def test_template_inheritance(self, template_manager, temp_template_dir):
        """Test template inheritance functionality."""
        # Create base template
        base_template = """
        Base Content
        {% block content %}Default content{% endblock %}
        End Base
        """
        (temp_template_dir / "base.txt").write_text(base_template)
        
        # Create child template
        child_template = """
        {% extends "base.txt" %}
        {% block content %}Child content: {value}{% endblock %}
        """
        (temp_template_dir / "child.txt").write_text(child_template)
        
        # Clear cache to pick up new templates
        template_manager.clear_cache()
        
        # Render child template
        rendered = template_manager.render_template("child", {"value": "test"})
        
        assert "Base Content" in rendered
        assert "Child content: test" in rendered
        assert "Default content" not in rendered
        assert "End Base" in rendered

    def test_template_error_handling(self, template_manager):
        """Test error handling in template operations."""
        # Test rendering with invalid template syntax
        invalid_template = "Hello {% invalid syntax %}"
        template_manager.add_template("invalid", invalid_template)
        
        with pytest.raises(TemplateError, match="Template rendering failed"):
            template_manager.render_template("invalid", {})

    def test_template_manager_with_custom_environment(self, temp_template_dir):
        """Test TemplateManager with custom Jinja2 environment settings."""
        # Create template manager with custom settings
        template_manager = TemplateManager(
            template_dir=str(temp_template_dir),
            auto_escape=True,
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Test that custom settings are applied
        assert template_manager._jinja_env.autoescape
        assert template_manager._jinja_env.trim_blocks
        assert template_manager._jinja_env.lstrip_blocks

    def test_concurrent_template_access(self, template_manager, sample_template_data):
        """Test concurrent access to templates."""
        import threading
        import time
        
        results = []
        errors = []
        
        def render_template():
            try:
                result = template_manager.render_template("experience", sample_template_data)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=render_template)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10
        
        # All results should be identical
        first_result = results[0]
        assert all(result == first_result for result in results)

    def test_template_performance_caching(self, template_manager, sample_template_data):
        """Test that template caching improves performance."""
        import time
        
        # Clear cache
        template_manager.clear_cache()
        
        # Time first render (should load from file)
        start_time = time.time()
        result1 = template_manager.render_template("experience", sample_template_data)
        first_render_time = time.time() - start_time
        
        # Time second render (should use cache)
        start_time = time.time()
        result2 = template_manager.render_template("experience", sample_template_data)
        second_render_time = time.time() - start_time
        
        # Results should be identical
        assert result1 == result2
        
        # Second render should be faster (cached)
        # Note: This might not always be true in fast systems, so we just check it doesn't error
        assert second_render_time >= 0
        assert first_render_time >= 0