"""Pytest configuration: put the repository root on sys.path so tests can
`import inference_pipeline`, `import utils`, etc. without installing a package.
"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
