#!/bin/bash

echo "Cleaning up old builds..."
rm -rf dist/ build/ *.egg-info

echo "Building the new distribution..."
python -m build

echo "Uploading to PyPI..."
python -m twine upload dist/* --skip-existing
