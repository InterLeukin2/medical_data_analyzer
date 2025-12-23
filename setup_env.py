"""
Virtual Environment Setup Script
===============================

This script automatically creates and sets up a virtual environment with all required dependencies.
"""

import os
import sys
import subprocess
import venv
from pathlib import Path


def setup_virtual_environment():
    """
    Creates and sets up the virtual environment with required dependencies
    """
    venv_dir = Path("myvenv")
    requirements_file = Path("medical_data_analyzer/requirements.txt")
    
    # Check if virtual environment already exists
    if venv_dir.exists():
        print(f"Virtual environment already exists at {venv_dir}")
        print("Skipping virtual environment creation...")
    else:
        print(f"Creating virtual environment at {venv_dir}")
        venv.create(venv_dir, with_pip=True)
        print("Virtual environment created successfully!")
    
    # Determine the path to the pip executable
    if sys.platform == "win32":
        pip_path = venv_dir / "Scripts" / "pip"
        python_path = venv_dir / "Scripts" / "python"
    else:
        pip_path = venv_dir / "bin" / "pip"
        python_path = venv_dir / "bin" / "python"
    
    # Upgrade pip first
    print("Upgrading pip...")
    subprocess.check_call([str(pip_path), "install", "--upgrade", "pip"])
    
    # Install requirements
    if requirements_file.exists():
        print(f"Installing dependencies from {requirements_file}...")
        subprocess.check_call([str(pip_path), "install", "-r", str(requirements_file)])
        print("Dependencies installed successfully!")
    else:
        print(f"Warning: {requirements_file} not found!")
        return False
    
    # Install additional packages that might be needed
    try:
        subprocess.check_call([str(pip_path), "install", "tkinter"])
    except subprocess.CalledProcessError:
        print("Note: tkinter might already be available in your Python installation or unavailable via pip")
    
    print(f"\nVirtual environment setup complete!")
    print(f"To activate the environment, use:")
    if sys.platform == "win32":
        print(f"  Windows: myvenv\\Scripts\\activate")
    else:
        print(f"  macOS/Linux: source myvenv/bin/activate")
    
    print(f"\nAfter activating the environment, you can run the analyzer with:")
    print(f"  python medical_data_analyzer/src/main.py <input_file>")
    
    return True


def main():
    """
    Main function to run the virtual environment setup
    """
    print("Setting up virtual environment for Medical Data Analyzer...")
    success = setup_virtual_environment()
    
    if success:
        print("\nSetup completed successfully!")
    else:
        print("\nSetup failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()