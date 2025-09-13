"""
OMR Package Setup Configuration
==============================

Setup configuration for the OMR (Optical Music Recognition) package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = [
        "opencv-python>=4.8.0",
        "numpy>=1.21.0",
        "ultralytics>=8.0.0",
        "streamlit>=1.28.0",
        "plotly>=5.15.0",
        "music21>=9.1.0",
        "lxml>=4.9.0",
        "albumentations>=1.3.0",
        "scikit-image>=0.21.0",
        "scipy>=1.11.0",
        "pillow>=10.0.0",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "tqdm>=4.66.0"
    ]

setup(
    name="omr-pipeline",
    version="1.0.0",
    author="OMR Development Team",
    author_email="omr@example.com",
    description="Comprehensive Optical Music Recognition Pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/omr-pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0"
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "sphinx-autodoc-typehints>=1.24.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "omr-pipeline=src.omr_pipeline:main",
            "omr-evaluate=src.evaluation.evaluation_cli:main",
            "omr-ui=src.ui.correction_interface:main",
        ],
    },
    package_data={
        "src": [
            "models/*.pt",
            "config/*.json",
            "templates/*.html"
        ]
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "optical music recognition",
        "omr",
        "sheet music",
        "musicxml",
        "computer vision",
        "deep learning",
        "yolo",
        "music information retrieval"
    ],
    project_urls={
        "Bug Reports": "https://github.com/example/omr-pipeline/issues",
        "Source": "https://github.com/example/omr-pipeline",
        "Documentation": "https://omr-pipeline.readthedocs.io/",
    },
)