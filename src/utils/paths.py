import os

# Get project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def get_project_path(relative_path):
    """Convert a path relative to project root into absolute path"""
    return os.path.join(PROJECT_ROOT, relative_path) 