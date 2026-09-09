#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import subprocess
from pathlib import Path

def install_requirements():
    """Install project dependencies before starting the server."""

    base_dir = Path(__file__).resolve().parent
    requirements_file = base_dir / "requirements.txt"

    if not requirements_file.exists():
        print("WARNING: requirements.txt was not found.")
        return

    print("\n" + "=" * 60)
    print("Checking project dependencies...")
    print("=" * 60)

    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                str(requirements_file),
                "--disable-pip-version-check",
            ]
        )

        print("\nDependencies are ready.\n")

    except subprocess.CalledProcessError as exc:
        print("\nERROR: Dependency installation failed.")
        print(
            "You can install them manually with:\n"
            "python -m pip install -r requirements.txt"
        )
        sys.exit(exc.returncode)


def main():
    """Run administrative tasks."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "pbl.settings")
    if (
        len(sys.argv) > 1
        and sys.argv[1] == "runserver"
        and os.environ.get("RUN_MAIN") != "true"
    ):
        install_requirements()
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()

    
