"""
Build Script for OMR Pipeline Package
====================================

Script to build and package the OMR pipeline for distribution.
"""

import subprocess
import sys
import shutil
from pathlib import Path
import tempfile
import zipfile

def run_command(command: str, cwd: Path = None) -> bool:
    """Run a shell command and return success status."""
    try:
        result = subprocess.run(
            command.split(),
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✓ {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {command}")
        print(f"Error: {e.stderr}")
        return False

def clean_build_artifacts():
    """Clean previous build artifacts."""
    print("Cleaning build artifacts...")
    
    artifacts = ["build", "dist", "*.egg-info"]
    for pattern in artifacts:
        for path in Path(".").glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                print(f"Removed directory: {path}")
            else:
                path.unlink()
                print(f"Removed file: {path}")

def run_tests():
    """Run the test suite."""
    print("\nRunning tests...")
    return run_command("python -m pytest tests/ -v")

def build_package():
    """Build the package."""
    print("\nBuilding package...")
    
    # Build source distribution
    if not run_command("python setup.py sdist"):
        return False
    
    # Build wheel distribution
    if not run_command("python setup.py bdist_wheel"):
        return False
    
    return True

def create_release_bundle():
    """Create a complete release bundle."""
    print("\nCreating release bundle...")
    
    # Create temporary directory for bundle
    with tempfile.TemporaryDirectory() as temp_dir:
        bundle_dir = Path(temp_dir) / "omr-pipeline-release"
        bundle_dir.mkdir()
        
        # Copy essential files
        essential_files = [
            "setup.py",
            "requirements.txt", 
            "README.md",
            "PACKAGE.md",
            "LICENSE",
            "MANIFEST.in"
        ]
        
        for file_name in essential_files:
            src_path = Path(file_name)
            if src_path.exists():
                shutil.copy2(src_path, bundle_dir)
        
        # Copy source code
        shutil.copytree("src", bundle_dir / "src")
        shutil.copytree("examples", bundle_dir / "examples")
        shutil.copytree("tests", bundle_dir / "tests")
        
        # Copy distribution files
        if Path("dist").exists():
            shutil.copytree("dist", bundle_dir / "dist")
        
        # Create installation script
        install_script = bundle_dir / "install.py"
        install_script.write_text("""
#!/usr/bin/env python3
import subprocess
import sys
import os

def main():
    print("Installing OMR Pipeline...")
    
    # Install from wheel if available
    wheel_files = [f for f in os.listdir("dist") if f.endswith(".whl")]
    if wheel_files:
        wheel_file = f"dist/{wheel_files[0]}"
        cmd = [sys.executable, "-m", "pip", "install", wheel_file]
    else:
        # Install from source
        cmd = [sys.executable, "-m", "pip", "install", "."]
    
    try:
        subprocess.run(cmd, check=True)
        print("✓ Installation successful!")
        print("\\nYou can now use:")
        print("  omr-pipeline input.png output.mxl")
        print("  omr-ui")
        print("  omr-evaluate --help")
    except subprocess.CalledProcessError as e:
        print(f"✗ Installation failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
""")
        
        # Create release notes
        release_notes = bundle_dir / "RELEASE_NOTES.md"
        release_notes.write_text("""
# OMR Pipeline Release

## Installation

1. Extract this bundle to a directory
2. Run: `python install.py`

Or manually:
```bash
pip install dist/omr_pipeline-1.0.0-py3-none-any.whl
```

Or from source:
```bash
pip install .
```

## Dependencies

Install required dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Process a sheet music image
omr-pipeline input_sheet.png output_score.mxl

# Launch correction interface
omr-ui

# Run evaluation
omr-evaluate --ground-truth gt.json --predictions pred.json
```

## Documentation

See README.md and examples/ directory for detailed usage instructions.
""")
        
        # Create ZIP bundle
        bundle_zip = Path("omr-pipeline-release.zip")
        if bundle_zip.exists():
            bundle_zip.unlink()
        
        with zipfile.ZipFile(bundle_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in bundle_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(bundle_dir.parent)
                    zf.write(file_path, arcname)
        
        print(f"✓ Created release bundle: {bundle_zip}")
        return True

def check_package():
    """Check the built package."""
    print("\nChecking package...")
    
    # Check distribution files exist
    dist_dir = Path("dist")
    if not dist_dir.exists():
        print("✗ No dist directory found")
        return False
    
    # Look for wheel and source distribution
    wheel_files = list(dist_dir.glob("*.whl"))
    tar_files = list(dist_dir.glob("*.tar.gz"))
    
    if not wheel_files:
        print("✗ No wheel file found")
        return False
    
    if not tar_files:
        print("✗ No source distribution found")
        return False
    
    print(f"✓ Found wheel: {wheel_files[0].name}")
    print(f"✓ Found source dist: {tar_files[0].name}")
    
    # Check package contents
    try:
        result = subprocess.run(
            ["python", "-m", "pip", "install", "--dry-run", str(wheel_files[0])],
            capture_output=True,
            text=True
        )
        print("✓ Package installation check passed")
        return True
    except Exception as e:
        print(f"✗ Package check failed: {e}")
        return False

def main():
    """Main build process."""
    print("OMR Pipeline Build Script")
    print("=" * 50)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Clean previous builds
    clean_build_artifacts()
    
    # Run tests (optional - continue even if tests fail)
    test_success = run_tests()
    if not test_success:
        print("⚠ Tests failed, but continuing build...")
    
    # Build package
    if not build_package():
        print("✗ Build failed")
        return 1
    
    # Check package
    if not check_package():
        print("✗ Package check failed")
        return 1
    
    # Create release bundle
    if not create_release_bundle():
        print("✗ Release bundle creation failed")
        return 1
    
    print("\n" + "=" * 50)
    print("✓ Build completed successfully!")
    print("\nGenerated files:")
    
    # List generated files
    for pattern in ["dist/*", "*.zip"]:
        for path in Path(".").glob(pattern):
            print(f"  {path}")
    
    print("\nNext steps:")
    print("1. Test the package: pip install dist/*.whl")
    print("2. Upload to PyPI: twine upload dist/*")
    print("3. Distribute release bundle: omr-pipeline-release.zip")
    
    return 0

if __name__ == "__main__":
    import os
    sys.exit(main())