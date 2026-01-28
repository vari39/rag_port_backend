#!/usr/bin/env python3
"""
Setup Script for Multilingual Processing System
Automates the installation and setup process
"""

import os
import sys
import subprocess
import platform

VENV_DIR = ".venv"


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"Running: {description}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"✅ {description} - Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed")
        if e.stderr:
            print(f"Error: {e.stderr}")
        else:
            print(f"Error: {e.stdout}")
        return False


def check_python_version():
    """Check Python version of the interpreter running this script"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False


def get_venv_python():
    """Return path to the virtualenv python."""
    system = platform.system().lower()
    if system == "windows":
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    else:
        return os.path.join(VENV_DIR, "bin", "python")


def install_system_dependencies():
    """Install system dependencies based on OS"""
    print("\nInstalling system dependencies...")

    system = platform.system().lower()

    if system == "darwin":  # macOS
        commands = [
            ("brew install tesseract", "Install Tesseract OCR (macOS)"),
            ("brew install libffi openssl", "Install additional dependencies (macOS)"),
        ]
    elif system == "linux":
        commands = [
            ("sudo apt-get update", "Update package list (Ubuntu)"),
            ("sudo apt-get install -y tesseract-ocr", "Install Tesseract OCR (Ubuntu)"),
            ("sudo apt-get install -y libtesseract-dev", "Install Tesseract development files (Ubuntu)"),
            ("sudo apt-get install -y libgl1-mesa-glx libglib2.0-0",
             "Install OpenCV dependencies (Ubuntu)"),
        ]
    else:
        print(f"❌ Unsupported operating system: {system}")
        return False

    success = True
    for command, description in commands:
        if not run_command(command, description):
            success = False

    return success


def create_virtual_environment():
    """Create virtual environment if it does not exist"""
    print("\nSetting up virtual environment...")

    if os.path.exists(VENV_DIR):
        print("✅ Virtual environment already exists")
        return True

    # Use the current interpreter to create the venv
    cmd = f'"{sys.executable}" -m venv "{VENV_DIR}"'
    if run_command(cmd, "Create virtual environment"):
        print("✅ Virtual environment created")
        print("\nTo manually activate the virtual environment, run:")
        print("  source .venv/bin/activate  # On macOS/Linux")
        print("  .venv\\Scripts\\activate    # On Windows")
        return True
    else:
        return False


def install_python_dependencies():
    """Install Python dependencies into the virtual environment"""
    print("\nInstalling Python dependencies...")

    venv_python = get_venv_python()
    if not os.path.exists(venv_python):
        print(f"❌ Could not find virtual environment Python at: {venv_python}")
        print("   Make sure the virtual environment was created successfully.")
        return False

    # Upgrade pip first
    if not run_command(f'"{venv_python}" -m pip install --upgrade pip', "Upgrade pip"):
        return False

    # Install requirements with specific version constraints
    commands = [
        (f'"{venv_python}" -m pip install "numpy>=1.24.3,<2.0.0"',
         "Install NumPy with version constraints"),
        (f'"{venv_python}" -m pip install "torch>=2.1.1,<2.6.0"',
         "Install PyTorch with version constraints"),
        (f'"{venv_python}" -m pip install "huggingface-hub>=0.16.0,<0.20.0"',
         "Install HuggingFace Hub with version constraints"),
        (f'"{venv_python}" -m pip install -r requirements.txt',
         "Install remaining Python dependencies"),
    ]

    success = True
    for command, description in commands:
        if not run_command(command, description):
            success = False

    if success:
        print("✅ Python dependencies installed")
        return True
    else:
        return False


def verify_installation():
    """Run verification script inside the virtual environment"""
    print("\nVerifying installation...")

    venv_python = get_venv_python()
    if not os.path.exists(venv_python):
        print(f"❌ Could not find virtual environment Python at: {venv_python}")
        return False

    cmd = f'"{venv_python}" verify_install.py'
    if run_command(cmd, "Run installation verification"):
        return True
    else:
        print("❌ Installation verification failed")
        return False


def main():
    """Main setup function"""
    print("Multilingual Processing System - Setup Script")
    print("=" * 50)

    # Check Python version of the interpreter running this script
    if not check_python_version():
        print("\n❌ Setup failed: Incompatible Python version")
        return False

    # Install system dependencies
    if not install_system_dependencies():
        print("\n❌ Setup failed: System dependencies installation failed")
        return False

    # Create virtual environment
    if not create_virtual_environment():
        print("\n❌ Setup failed: Virtual environment creation failed")
        return False

    # Install Python dependencies into .venv
    if not install_python_dependencies():
        print("\n❌ Setup failed: Python dependencies installation failed")
        return False

    # Verify installation (using .venv Python)
    if not verify_installation():
        print("\n❌ Setup failed: Installation verification failed")
        return False

    # Success
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Activate virtual environment (optional if you just want to run tools manually):")
    print("   source .venv/bin/activate      # macOS/Linux")
    print("   .venv\\Scripts\\activate        # Windows")
    print("\n2. Run the demo:")
    print("   python multilingual_demo.py     # after activating .venv")
    print("\n3. Run tests:")
    print("   python -m pytest tests/test_multilingual_processing.py -v")
    print("\n4. Read the documentation:")
    print("   See README.md for detailed usage instructions")

    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        sys.exit(1)
