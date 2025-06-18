#!/usr/bin/env python3
"""Script to migrate existing log files to the new logging structure."""

import os
import shutil
from pathlib import Path
from datetime import datetime


def migrate_logs():
    """Migrate existing log files to the new logs directory structure."""
    project_root = Path(__file__).parent.parent
    logs_dir = project_root / "logs"

    # Ensure log directories exist
    (logs_dir / "debug").mkdir(parents=True, exist_ok=True)
    (logs_dir / "error").mkdir(parents=True, exist_ok=True)
    (logs_dir / "access").mkdir(parents=True, exist_ok=True)

    # List of old log files that might exist in the root
    old_log_files = [
        "debug.log",
        "error.log",
        "app.log",
        "workflow_test.log",
        "test_pipeline.log",
        "test_ai_engineer_content.log",
        "content_writer_test.log",
        "parser_test.log",
    ]

    migrated_files = []

    for log_file in old_log_files:
        old_path = project_root / log_file
        if old_path.exists():
            # Determine destination based on file type
            if "test" in log_file or "workflow" in log_file:
                new_path = logs_dir / "debug" / log_file
            elif "error" in log_file:
                new_path = logs_dir / "error" / log_file
            else:
                new_path = logs_dir / "debug" / log_file

            # Create backup with timestamp if destination exists
            if new_path.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"{new_path.stem}_{timestamp}{new_path.suffix}"
                backup_path = new_path.parent / backup_name
                shutil.move(str(new_path), str(backup_path))
                print(f"Backed up existing {new_path.name} to {backup_name}")

            # Move the file
            shutil.move(str(old_path), str(new_path))
            migrated_files.append((log_file, str(new_path.relative_to(project_root))))
            print(f"Migrated {log_file} -> {new_path.relative_to(project_root)}")

    if migrated_files:
        print(f"\nSuccessfully migrated {len(migrated_files)} log files:")
        for old_name, new_path in migrated_files:
            print(f"  {old_name} -> {new_path}")
    else:
        print("No log files found to migrate.")

    print("\nLog migration completed!")
    print("\nNew logging structure:")
    print("  logs/debug/     - Debug and test logs")
    print("  logs/error/     - Error and critical logs")
    print("  logs/access/    - HTTP access logs")
    print("\nAll future logs will be created in the appropriate subdirectories.")


if __name__ == "__main__":
    migrate_logs()
