"""Runs every synthetic test file in this folder. From the project root:  python tests/run_all.py"""
import subprocess
import sys
from pathlib import Path

here = Path(__file__).parent
failed = []
for test in sorted(here.glob("test_*.py")):
    print(f"\n=== {test.name}")
    if subprocess.call([sys.executable, str(test)]) != 0:
        failed.append(test.name)
print("\nALL PASSED" if not failed else f"\nFAILED: {', '.join(failed)}")
sys.exit(1 if failed else 0)
