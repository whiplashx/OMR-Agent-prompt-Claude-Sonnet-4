# OMR Pipeline Usage Guide
Complete guide to using the Optical Music Recognition Pipeline

## 🚀 Quick Start

### 1. Installation

#### Option A: Install from Package (when published)
```bash
pip install omr-pipeline
```

#### Option B: Install from Source (current setup)
```bash
# Clone/download the OMR directory to your machine
cd d:\OMR

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### 2. Verify Installation
```bash
# Test if the main module loads
python -c "from src.omr_pipeline import OMRPipeline; print('✓ OMR Pipeline ready!')"
```

## 📋 System Requirements

- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB for dependencies + model files
- **OS**: Windows, macOS, or Linux

## 🎼 Basic Usage

### Method 1: Python API (Recommended)

```python
from src.omr_pipeline import OMRPipeline
import numpy as np
from PIL import Image

# Initialize the pipeline
pipeline = OMRPipeline()

# Process a sheet music image
result = pipeline.process_image(
    image_path="path/to/your/sheet_music.png",
    output_path="output/score.mxl"
)

# Check results
print(f"✓ MusicXML saved to: {result['musicxml_path']}")
print(f"✓ Quality score: {result['quality_score']:.2f}")
print(f"✓ Processing time: {result['processing_time']:.1f}s")

# Access detailed confidence data
confidence = result['confidence_data']
print(f"Overall confidence: {confidence['overall_confidence']:.2f}")
```

### Method 2: Command Line Interface

```bash
# Basic processing
python src/omr_pipeline.py input_image.png output_score.mxl

# With custom confidence threshold
python src/omr_pipeline.py input_image.png output_score.mxl --confidence-threshold 0.8

# Batch processing
python src/omr_pipeline.py input_folder/ output_folder/ --batch

# Get help
python src/omr_pipeline.py --help
```

### Method 3: Interactive UI

```bash
# Launch the manual correction interface
streamlit run src/ui/correction_interface.py

# Or use the shortcut (if installed as package)
# omr-ui
```

## 📁 File Formats

### Input Formats Supported
- **Images**: PNG, JPG, JPEG, TIFF, BMP
- **Recommended**: High-resolution PNG files (300+ DPI)
- **Size**: Any size (automatically resized if needed)

### Output Formats
- **MusicXML**: `.mxl` files (MuseScore compatible)
- **JSON**: Confidence and metadata reports
- **Images**: Annotated detection results (optional)

## 🛠 Advanced Usage

### Custom Configuration

```python
# Custom configuration
config = {
    'confidence_threshold': 0.7,        # Symbol detection threshold
    'staff_line_thickness': 2,          # Expected staff line thickness
    'min_staff_length': 100,            # Minimum staff line length
    'divisions': 480,                   # MusicXML time divisions
    'output_format': 'musicxml',        # Output format
    'apply_skew_correction': True,      # Enable skew correction
    'apply_denoising': True,            # Enable noise reduction
}

pipeline = OMRPipeline(config=config)
```

### Batch Processing

```python
# Process multiple images
image_paths = [
    "sheet1.png",
    "sheet2.jpg", 
    "sheet3.tiff"
]

results = pipeline.process_batch(
    image_paths=image_paths,
    output_dir="batch_output/",
    parallel=True  # Process in parallel
)

# Check results for each image
for i, result in enumerate(results):
    if result:
        print(f"✓ Image {i+1}: {result['quality_score']:.2f}")
    else:
        print(f"✗ Image {i+1}: Processing failed")
```

### Working with Results

```python
result = pipeline.process_image("sheet_music.png")

# Access MusicXML content
musicxml_path = result['musicxml_path']
with open(musicxml_path, 'r') as f:
    musicxml_content = f.read()

# Access confidence data
confidence_data = result['confidence_data']
low_confidence_symbols = confidence_data['low_confidence_symbols']

# Quality assessment
quality_score = result['quality_score']
if quality_score < 0.5:
    print("⚠️ Low quality result - consider manual correction")
elif quality_score > 0.8:
    print("✓ High quality result")
```

## 🎯 Specific Use Cases

### 1. Scanning Physical Sheet Music

```python
# Best practices for scanned sheet music
config = {
    'apply_denoising': True,           # Remove scanner artifacts
    'apply_skew_correction': True,     # Correct scanning angle
    'confidence_threshold': 0.6,      # Lower threshold for scanned images
    'staff_line_thickness': 3,        # Thicker lines in scans
}

pipeline = OMRPipeline(config=config)
result = pipeline.process_image("scanned_sheet.png")
```

### 2. Digital Sheet Music (PDF exports)

```python
# Optimized for clean digital images
config = {
    'apply_denoising': False,          # Skip denoising for clean images
    'confidence_threshold': 0.8,       # Higher threshold for clean images
    'staff_line_thickness': 1,         # Thinner lines in digital
}

pipeline = OMRPipeline(config=config)
result = pipeline.process_image("digital_sheet.png")
```

### 3. Handwritten Music (Limited Support)

```python
# Experimental support for handwritten music
config = {
    'confidence_threshold': 0.4,       # Much lower threshold
    'apply_denoising': True,
    'model_type': 'handwritten',       # If available
}

pipeline = OMRPipeline(config=config)
result = pipeline.process_image("handwritten_sheet.png")

# Likely needs manual correction
if result['quality_score'] < 0.7:
    print("Manual correction recommended for handwritten music")
```

## 🔧 Manual Correction Workflow

1. **Process with Pipeline**:
```python
result = pipeline.process_image("sheet.png", "output.mxl")
```

2. **Check Quality**:
```python
if result['quality_score'] < 0.7:
    print("Consider manual correction")
```

3. **Launch Correction UI**:
```bash
streamlit run src/ui/correction_interface.py
```

4. **In the UI**:
   - Load the processed result
   - Review detected symbols
   - Click to correct misidentified symbols
   - Add missing symbols
   - Export corrected MusicXML

## 📊 Evaluation and Testing

### Evaluate Pipeline Performance

```python
from src.evaluation.metrics import OMREvaluator

evaluator = OMREvaluator()

# Compare results against ground truth
metrics = evaluator.evaluate_predictions(
    predictions_file="predictions.json",
    ground_truth_file="ground_truth.json"
)

print(f"Symbol detection F1: {metrics['symbol_f1']:.3f}")
print(f"Pitch accuracy: {metrics['pitch_accuracy']:.3f}")
```

### Generate Test Data

```python
from src.data.dataset_manager import DatasetManager

dataset_manager = DatasetManager()

# Generate synthetic training data
dataset_manager.generate_synthetic_data(
    output_dir="synthetic_data/",
    num_samples=1000,
    image_size=(800, 1200)
)
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# If you get "module not found" errors
pip install -r requirements.txt

# For development installation
pip install -e .
```

#### 2. Memory Issues
```python
# For large images, reduce resolution
config = {'max_image_size': (1600, 1200)}
pipeline = OMRPipeline(config=config)
```

#### 3. Low Quality Results
```python
# Try adjusting confidence threshold
config = {'confidence_threshold': 0.5}  # Lower = more detections

# Enable all preprocessing
config = {
    'apply_denoising': True,
    'apply_skew_correction': True,
    'apply_contrast_enhancement': True
}
```

#### 4. Model Loading Issues
```python
# Check if models are available
import os
model_path = os.environ.get('OMR_MODEL_PATH', 'models/')
print(f"Looking for models in: {model_path}")

# Download pre-trained models (if available)
# pipeline.download_models()  # Future feature
```

### Getting Help

1. **Check the logs**: The pipeline provides detailed logging
2. **Review examples**: Check the `examples/` directory
3. **Run tests**: Use `pytest tests/` to verify installation
4. **Check configuration**: Verify your config parameters

## 📈 Performance Tips

### Optimize Processing Speed
```python
# For faster processing
config = {
    'apply_denoising': False,      # Skip if image is clean
    'max_image_size': (1200, 800), # Smaller size = faster
    'parallel_processing': True,    # Use multiple cores
}
```

### Optimize Accuracy
```python
# For better accuracy
config = {
    'confidence_threshold': 0.6,    # Lower threshold
    'apply_denoising': True,        # Clean the image
    'apply_skew_correction': True,  # Fix rotation
    'staff_line_thickness': 2,      # Match your images
}
```

### Memory Management
```python
# For large batch processing
pipeline = OMRPipeline()

for image_path in large_image_list:
    result = pipeline.process_image(image_path)
    # Process result immediately
    # Don't accumulate results in memory
```

## 🎯 Next Steps

After getting familiar with basic usage:

1. **Explore Examples**: Check `examples/` directory for specific scenarios
2. **Try Manual Correction**: Use the Streamlit UI for complex scores
3. **Evaluate Performance**: Use the evaluation tools on your data
4. **Customize Configuration**: Tune parameters for your specific use case
5. **Contribute**: Help improve the pipeline with feedback and contributions

---

**Need help?** Check the documentation in `README.md` or explore the working examples in the `examples/` directory!