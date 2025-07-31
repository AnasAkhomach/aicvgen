#!/usr/bin/env python3
"""Script to fix Pydantic V2 deprecation warnings by replacing Config classes with model_config."""

import re


def fix_pydantic_config(file_path: str) -> None:
    """Fix Pydantic Config classes in the given file."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Pattern 1: Config class with json_encoders and arbitrary_types_allowed
    pattern1 = re.compile(
        r"    class Config:\s*\n"
        r'        """[^"]*"""\s*\n\s*\n'
        r"        arbitrary_types_allowed = True\s*\n"
        r"        json_encoders = \{\s*\n"
        r"            datetime: lambda v: v\.isoformat\(\) if v else None,\s*\n"
        r"        \}",
        re.MULTILINE,
    )

    replacement1 = (
        "    model_config = ConfigDict(\n"
        "        arbitrary_types_allowed=True,\n"
        "        json_encoders={\n"
        "            datetime: lambda v: v.isoformat() if v else None,\n"
        "        }\n"
        "    )"
    )

    # Pattern 2: Simple Config class with json_encoders
    pattern2 = re.compile(
        r"    class Config:\s*\n"
        r"        arbitrary_types_allowed = True\s*\n"
        r"        json_encoders = \{[^}]*\}",
        re.MULTILINE,
    )

    # Apply replacements
    content = pattern1.sub(replacement1, content)

    # Handle remaining Config classes manually
    remaining_configs = re.findall(
        r"    class Config:[^}]*}", content, re.MULTILINE | re.DOTALL
    )

    print(f"Found {len(remaining_configs)} remaining Config classes to fix manually")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Fixed Pydantic Config classes in {file_path}")


if __name__ == "__main__":
    fix_pydantic_config("src/models/agent_output_models.py")
