#!/usr/bin/env python3
"""
format.py

Format all Python code and sort imports using Ruff (also in notebooks via nbqa).
"""

import subprocess
import sys


def run(cmd):
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: exited with code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)

def main():
    # 1) Autofix lint errors (including import sorting) in .py och .ipynb
    run(["ruff", "check", "--fix", "."])
    run(["nbqa", "ruff", "check", "--fix", "."])
    # 2) Kör nbqa+black för styckeindrag och radbrytningar
    run(["nbqa", "black", "."])
    print("✔ Formatting and import sorting completed for both .py and .ipynb")

if __name__ == "__main__":
    main()
