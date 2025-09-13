# OMR Pipeline

A comprehensive **Optical Music Recognition (OMR)** package for converting sheet music images into structured digital formats.

## Package Installation

### Using pip (recommended)

```bash
pip install omr-pipeline
```

### From source

```bash
git clone https://github.com/example/omr-pipeline.git
cd omr-pipeline
pip install -e .
```

### Development installation

```bash
git clone https://github.com/example/omr-pipeline.git
cd omr-pipeline
pip install -e ".[dev,docs]"
```

## Quick Start

### Command Line Interface

After installation, you can use the OMR pipeline directly from the command line:

```bash
# Process a single image
omr-pipeline input_sheet.png output_score.mxl

# Process a batch of images
omr-pipeline input_folder/ output_folder/ --batch

# Launch the correction interface
omr-ui

# Evaluate the pipeline
omr-evaluate --ground-truth gt.json --predictions pred.json
```

### Python API

```python
from omr_pipeline import OMRPipeline

# Initialize the pipeline
pipeline = OMRPipeline()

# Process an image
result = pipeline.process_image("sheet_music.png")

# Access results
musicxml_path = result.musicxml_path
confidence_data = result.confidence_data
quality_score = result.quality_score

print(f"Generated MusicXML: {musicxml_path}")
print(f"Overall confidence: {quality_score:.2f}")
```

## Package Structure

```
omr-pipeline/
├── src/
│   ├── __init__.py
│   ├── omr_pipeline.py          # Main pipeline orchestrator
│   ├── preprocessing/           # Image preprocessing
│   ├── detection/              # Staff and symbol detection  
│   ├── reconstruction/         # Music reconstruction
│   ├── output/                 # MusicXML and JSON generation
│   ├── ui/                     # Manual correction interface
│   ├── data/                   # Dataset management
│   └── evaluation/             # Evaluation metrics
├── examples/                   # Usage examples
├── tests/                      # Unit tests
├── docs/                       # Documentation
├── setup.py                    # Package setup
├── requirements.txt            # Dependencies
└── README.md                  # This file
```

## Core Features

- **Image Preprocessing**: Noise reduction, binarization, skew correction
- **Staff Detection**: Robust staff line detection and removal
- **Symbol Detection**: 50+ musical symbol types using YOLO
- **Music Reconstruction**: Pitch calculation, rhythm analysis, voice separation
- **MusicXML Output**: MuseScore-compatible format (divisions=480)
- **Confidence Reporting**: JSON output with detection confidences
- **Manual Correction**: Interactive Streamlit interface
- **Evaluation Metrics**: Comprehensive assessment tools
- **Dataset Management**: Synthetic data generation and augmentation

## Configuration

The package can be configured through environment variables or configuration files:

```python
import os
from omr_pipeline import OMRPipeline

# Configure via environment
os.environ['OMR_MODEL_PATH'] = '/path/to/models'
os.environ['OMR_CONFIDENCE_THRESHOLD'] = '0.7'

# Or via configuration
config = {
    'model_path': '/path/to/models',
    'confidence_threshold': 0.7,
    'output_format': 'musicxml'
}

pipeline = OMRPipeline(config=config)
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_preprocessing.py
pytest tests/test_detection.py
pytest tests/test_reconstruction.py

# Run with coverage
pytest --cov=src tests/
```

## Documentation

Build the documentation:

```bash
cd docs/
make html
```

View documentation at `docs/_build/html/index.html`

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/example/omr-pipeline/issues)
- **Documentation**: [Read the Docs](https://omr-pipeline.readthedocs.io/)
- **Discussions**: [GitHub Discussions](https://github.com/example/omr-pipeline/discussions)

## Citation

If you use this package in your research, please cite:

```bibtex
@software{omr_pipeline,
  title={OMR Pipeline: Comprehensive Optical Music Recognition},
  author={OMR Development Team},
  year={2024},
  url={https://github.com/example/omr-pipeline}
}
```