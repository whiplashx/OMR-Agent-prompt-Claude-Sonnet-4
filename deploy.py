"""
Deployment Script for OMR Pipeline
=================================

Script to deploy the OMR pipeline package to various platforms.
"""

import subprocess
import sys
import os
from pathlib import Path
import json
import requests
import getpass

def run_command(command: str, cwd: Path = None, capture_output: bool = True) -> tuple:
    """Run a shell command and return success status and output."""
    try:
        if isinstance(command, str):
            command = command.split()
        
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=capture_output,
            text=True,
            check=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def check_dependencies():
    """Check if required deployment tools are installed."""
    print("Checking deployment dependencies...")
    
    required_tools = ["twine", "git"]
    missing_tools = []
    
    for tool in required_tools:
        success, _ = run_command(f"{tool} --version")
        if success:
            print(f"✓ {tool} is available")
        else:
            missing_tools.append(tool)
            print(f"✗ {tool} is missing")
    
    if missing_tools:
        print(f"\nPlease install missing tools:")
        for tool in missing_tools:
            if tool == "twine":
                print(f"  pip install {tool}")
            else:
                print(f"  Install {tool} from your package manager")
        return False
    
    return True

def verify_package():
    """Verify package is ready for deployment."""
    print("\nVerifying package...")
    
    # Check if dist directory exists and has files
    dist_dir = Path("dist")
    if not dist_dir.exists():
        print("✗ No dist directory found. Run build.py first.")
        return False
    
    wheel_files = list(dist_dir.glob("*.whl"))
    tar_files = list(dist_dir.glob("*.tar.gz"))
    
    if not wheel_files or not tar_files:
        print("✗ Missing distribution files. Run build.py first.")
        return False
    
    print(f"✓ Found {len(wheel_files)} wheel file(s)")
    print(f"✓ Found {len(tar_files)} source distribution(s)")
    
    # Check package metadata
    success, output = run_command("twine check dist/*")
    if success:
        print("✓ Package metadata is valid")
        return True
    else:
        print(f"✗ Package validation failed: {output}")
        return False

def deploy_to_test_pypi():
    """Deploy package to Test PyPI."""
    print("\nDeploying to Test PyPI...")
    
    print("You will need Test PyPI credentials.")
    print("Get token from: https://test.pypi.org/manage/account/token/")
    
    # Upload to Test PyPI
    success, output = run_command([
        "twine", "upload", 
        "--repository", "testpypi",
        "--verbose",
        "dist/*"
    ], capture_output=False)
    
    if success:
        print("✓ Successfully uploaded to Test PyPI")
        print("Test installation with:")
        print("  pip install --index-url https://test.pypi.org/simple/ omr-pipeline")
        return True
    else:
        print(f"✗ Upload to Test PyPI failed")
        return False

def deploy_to_pypi():
    """Deploy package to production PyPI."""
    print("\nDeploying to production PyPI...")
    
    # Confirm deployment
    confirm = input("Are you sure you want to deploy to production PyPI? (yes/no): ")
    if confirm.lower() != "yes":
        print("Deployment cancelled.")
        return False
    
    print("You will need PyPI credentials.")
    print("Get token from: https://pypi.org/manage/account/token/")
    
    # Upload to PyPI
    success, output = run_command([
        "twine", "upload",
        "--verbose", 
        "dist/*"
    ], capture_output=False)
    
    if success:
        print("✓ Successfully uploaded to PyPI")
        print("Install with: pip install omr-pipeline")
        return True
    else:
        print(f"✗ Upload to PyPI failed")
        return False

def create_github_release():
    """Create a GitHub release."""
    print("\nCreating GitHub release...")
    
    # Check if we're in a git repository
    success, _ = run_command("git status")
    if not success:
        print("✗ Not in a git repository")
        return False
    
    # Get version from setup.py
    version = "v1.0.0"  # Default version
    
    # Create git tag
    tag_name = version
    success, _ = run_command(f"git tag {tag_name}")
    if not success:
        print(f"Tag {tag_name} may already exist")
    
    # Push tag
    success, _ = run_command(f"git push origin {tag_name}")
    if success:
        print(f"✓ Created and pushed tag {tag_name}")
    else:
        print(f"✗ Failed to push tag {tag_name}")
    
    print(f"Create release manually at: https://github.com/your-username/omr-pipeline/releases/new")
    print(f"Tag: {tag_name}")
    print("Attach the release bundle: omr-pipeline-release.zip")
    
    return True

def deploy_docker():
    """Create Docker deployment instructions."""
    print("\nCreating Docker deployment files...")
    
    # Create Dockerfile
    dockerfile_content = '''
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    libglib2.0-0 \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY setup.py .
COPY MANIFEST.in .

# Install the package
RUN pip install .

# Create non-root user
RUN useradd --create-home --shell /bin/bash omr
USER omr

# Set environment variables
ENV PYTHONPATH=/app
ENV OMR_MODEL_PATH=/app/models

# Expose port for Streamlit UI
EXPOSE 8501

# Default command
CMD ["omr-ui", "--server.port=8501", "--server.address=0.0.0.0"]
'''
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    # Create docker-compose.yml
    compose_content = '''
version: '3.8'

services:
  omr-pipeline:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./input:/app/input
      - ./output:/app/output
      - ./models:/app/models
    environment:
      - OMR_MODEL_PATH=/app/models
    restart: unless-stopped

  omr-api:
    build: .
    command: ["python", "-m", "src.api.server"]
    ports:
      - "8000:8000"
    volumes:
      - ./input:/app/input
      - ./output:/app/output
      - ./models:/app/models
    environment:
      - OMR_MODEL_PATH=/app/models
    restart: unless-stopped
'''
    
    with open("docker-compose.yml", "w") as f:
        f.write(compose_content)
    
    # Create deployment instructions
    docker_readme = '''
# Docker Deployment

## Building and Running

### Using Docker
```bash
# Build the image
docker build -t omr-pipeline .

# Run the UI
docker run -p 8501:8501 -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output omr-pipeline

# Run command line processing
docker run -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output omr-pipeline omr-pipeline /app/input/sheet.png /app/output/score.mxl
```

### Using Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Services

- **omr-pipeline**: Streamlit UI on port 8501
- **omr-api**: REST API on port 8000 (if implemented)

## Volumes

- `./input`: Input sheet music images
- `./output`: Generated MusicXML files
- `./models`: Pre-trained models

## Environment Variables

- `OMR_MODEL_PATH`: Path to model files
- `OMR_CONFIDENCE_THRESHOLD`: Detection confidence threshold
'''
    
    with open("DOCKER.md", "w") as f:
        f.write(docker_readme)
    
    print("✓ Created Docker deployment files:")
    print("  - Dockerfile")
    print("  - docker-compose.yml") 
    print("  - DOCKER.md")
    
    return True

def main():
    """Main deployment process."""
    print("OMR Pipeline Deployment Script")
    print("=" * 50)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Verify package
    if not verify_package():
        return 1
    
    # Deployment options
    print("\nDeployment options:")
    print("1. Deploy to Test PyPI")
    print("2. Deploy to production PyPI")
    print("3. Create GitHub release")
    print("4. Create Docker deployment files")
    print("5. All of the above")
    print("0. Exit")
    
    while True:
        choice = input("\nSelect deployment option (0-5): ").strip()
        
        if choice == "0":
            print("Deployment cancelled.")
            return 0
        
        elif choice == "1":
            deploy_to_test_pypi()
            break
        
        elif choice == "2":
            deploy_to_pypi()
            break
        
        elif choice == "3":
            create_github_release()
            break
        
        elif choice == "4":
            deploy_docker()
            break
        
        elif choice == "5":
            print("\nRunning full deployment...")
            deploy_to_test_pypi()
            input("\\nTest the Test PyPI installation, then press Enter to continue...")
            deploy_to_pypi()
            create_github_release()
            deploy_docker()
            break
        
        else:
            print("Invalid choice. Please select 0-5.")
    
    print("\n" + "=" * 50)
    print("✓ Deployment process completed!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())