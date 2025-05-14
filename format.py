#!/usr/bin/env python3
"""
format.py

Format all Python code and sort imports using Ruff.

Usage:
    python format.py
"""

import subprocess
import sys


def run(cmd):
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Ruff exited with code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)


def main():
    # 1) Autofix lint errors (including import sorting)
    run(["ruff", "check", "--fix", "."])
    # 2) Run the formatter to wrap lines and apply style rules
    run(["ruff", "format", "."])
    print("✔ Formatting and import sorting completed.")


if __name__ == "__main__":
    main()
