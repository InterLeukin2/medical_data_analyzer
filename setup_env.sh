#!/bin/bash

# Virtual Environment Setup Script
# ================================
#
# This script automatically creates and sets up a virtual environment with all required dependencies.

set -e  # Exit immediately if a command exits with a non-zero status

echo "Setting up virtual environment for Medical Data Analyzer..."

# Define variables
VENV_DIR="myvenv"
REQUIREMENTS_FILE="medical_data_analyzer/requirements.txt"

# Check if virtual environment already exists
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    echo "Skipping virtual environment creation..."
else
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created successfully!"
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Installing dependencies from $REQUIREMENTS_FILE..."
    pip install -r "$REQUIREMENTS_FILE"
    echo "Dependencies installed successfully!"
else
    echo "Warning: $REQUIREMENTS_FILE not found!"
    exit 1
fi

# Install additional packages that might be needed
pip install tkinter || echo "Note: tkinter might already be available in your Python installation"

echo
echo "Virtual environment setup complete!"
echo "To activate the environment in the future, use:"
echo "  source $VENV_DIR/bin/activate"
echo
echo "After activating the environment, you can run the analyzer with:"
echo "  python medical_data_analyzer/src/main.py <input_file>"

echo
echo "Setup completed successfully!"