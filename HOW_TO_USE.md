# 🎼 How to Use the OMR Pipeline
**Complete Guide to Getting Started with Optical Music Recognition**

## 🚀 Quick Start (5 Minutes)

### Step 1: Basic Setup
```bash
# Navigate to the OMR directory
cd d:\OMR

# Make sure dependencies are installed (you've already done this!)
# D:/OMR/.venv/Scripts/python.exe -m pip install -r requirements.txt
```

### Step 2: Test Your Installation
```bash
# Run the test script to verify everything works
D:/OMR/.venv/Scripts/python.exe test_installation.py
```

### Step 3: Try the Simple Example
```bash
# Run a working example with synthetic sheet music
D:/OMR/.venv/Scripts/python.exe simple_example.py
```

### Step 4: Process Real Sheet Music
```python
# Create a new Python file (e.g., my_omr_test.py)
import sys
sys.path.append('src')
from omr_pipeline import OMRPipeline
import cv2

# Load your sheet music image
image = cv2.imread('your_sheet_music.png')

# Create and run the pipeline
pipeline = OMRPipeline()
result = pipeline.process_image(image=image, image_path='your_sheet_music.png')

if result.success:
    print(f"✅ Success! Processing took {result.total_time:.2f} seconds")
    print(f"📊 Detected {len(result.detected_symbols or [])} symbols")
else:
    print(f"❌ Error: {result.error_message}")
```

## 📖 Detailed Usage Methods

### 🎯 Method 1: Python API (Recommended)

This is the most flexible way to use the OMR pipeline:

```python
import sys
sys.path.append('src')  # Add src directory to Python path
from omr_pipeline import OMRPipeline
import cv2
import numpy as np

# 1. Create pipeline instance
pipeline = OMRPipeline()

# 2. Load your sheet music image
image = cv2.imread('path/to/your/sheet_music.png')
# OR create from numpy array
# image = np.array(your_image_data)

# 3. Process the image
result = pipeline.process_image(
    image=image,
    image_path='sheet_music.png'  # Optional: for logging/output naming
)

# 4. Check results
if result.success:
    print("🎼 OMR Processing Results:")
    print(f"   ✅ Total time: {result.total_time:.2f} seconds")
    print(f"   📈 Preprocessing: {result.preprocessing_time:.2f}s")
    print(f"   🎵 Staff detection: {result.staff_detection_time:.2f}s") 
    print(f"   🔍 Symbol detection: {result.symbol_detection_time:.2f}s")
    print(f"   🎼 Music reconstruction: {result.music_reconstruction_time:.2f}s")
    print(f"   📄 Output generation: {result.output_generation_time:.2f}s")
    
    # Access detected elements
    if result.detected_staves:
        print(f"   📊 Found {len(result.detected_staves)} staff systems")
    
    if result.detected_symbols:
        print(f"   🎵 Detected {len(result.detected_symbols)} musical symbols")
        # Show first few symbols
        for i, symbol in enumerate(result.detected_symbols[:3]):
            print(f"      Symbol {i+1}: {symbol.get('class', 'unknown')}")
    
    if result.musical_elements:
        print(f"   🎼 Reconstructed musical elements available")
        
else:
    print(f"❌ Processing failed: {result.error_message}")
```

### 🌐 Method 2: Web Interface (Great for Manual Correction)

Launch the interactive web interface:

```bash
# Start the Streamlit web app
D:/OMR/.venv/Scripts/python.exe -m streamlit run src/ui/correction_interface.py
```

Then open your browser to `http://localhost:8501` and:

1. **Upload** your sheet music image
2. **Review** the automatically detected symbols
3. **Click** on symbols to correct them
4. **Add** missing symbols by clicking on the image
5. **Export** the corrected MusicXML file

### 🔧 Method 3: Advanced Configuration

Customize the pipeline for different types of sheet music:

```python
# Configuration for scanned sheet music
scanned_config = {
    'apply_denoising': True,           # Remove scanner noise
    'apply_skew_correction': True,     # Fix scanning angle
    'confidence_threshold': 0.6,       # Lower threshold for noisy images
    'staff_line_thickness': 3,         # Thicker lines in scans
}

# Configuration for clean digital images  
digital_config = {
    'apply_denoising': False,          # Skip denoising for clean images
    'confidence_threshold': 0.8,       # Higher threshold for clean images
    'staff_line_thickness': 1,         # Thinner lines in digital images
}

# Configuration for handwritten music (experimental)
handwritten_config = {
    'confidence_threshold': 0.4,       # Much lower threshold
    'apply_denoising': True,           # Clean up handwriting artifacts
}

# Create pipeline with custom config
pipeline = OMRPipeline(config=scanned_config)
```

### 📁 Method 4: Batch Processing

Process multiple sheet music files at once:

```python
import glob
from pathlib import Path

# Find all images in a directory
image_files = glob.glob("sheet_music_folder/*.png")
print(f"Found {len(image_files)} images to process")

pipeline = OMRPipeline()

# Process each image
results = []
for image_path in image_files:
    print(f"Processing: {Path(image_path).name}")
    
    # Load and process image
    image = cv2.imread(image_path)
    result = pipeline.process_image(image=image, image_path=image_path)
    
    results.append({
        'file': Path(image_path).name,
        'success': result.success,
        'time': result.total_time,
        'symbols': len(result.detected_symbols or [])
    })

# Summary report
print("\n📊 Batch Processing Results:")
for r in results:
    status = "✅" if r['success'] else "❌"
    print(f"   {status} {r['file']}: {r['symbols']} symbols, {r['time']:.1f}s")
```

## 📝 Working with Results

### Understanding the Output

The pipeline returns an `OMRResults` object with these key properties:

```python
result = pipeline.process_image(image, "sheet.png")

# Basic status
result.success                    # True if processing succeeded
result.error_message             # Error details if failed
result.total_time                # Total processing time

# Timing breakdown
result.preprocessing_time        # Image preprocessing time
result.staff_detection_time      # Staff detection time  
result.symbol_detection_time     # Symbol detection time
result.music_reconstruction_time # Music reconstruction time
result.output_generation_time    # MusicXML generation time

# Intermediate results
result.preprocessed_image        # Cleaned/processed image
result.detected_staves          # Staff line information
result.detected_symbols         # Raw symbol detections
result.musical_elements         # Reconstructed musical meaning

# Paths to generated files
result.musicxml_path           # Path to generated MusicXML file
result.confidence_json_path    # Path to confidence analysis JSON
```

### Accessing Detected Symbols

```python
if result.detected_symbols:
    for symbol in result.detected_symbols:
        symbol_type = symbol.get('class', 'unknown')
        confidence = symbol.get('confidence', 0)
        bbox = symbol.get('bbox', [0,0,0,0])  # [x1, y1, x2, y2]
        
        print(f"Symbol: {symbol_type}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Position: {bbox}")
```

### Working with Musical Elements

```python
if result.musical_elements:
    measures = result.musical_elements.get('measures', [])
    voices = result.musical_elements.get('voices', [])
    
    print(f"Found {len(measures)} measures")
    print(f"Found {len(voices)} voices")
    
    # Access individual musical elements
    for measure in measures[:3]:  # First 3 measures
        elements = measure.get('elements', [])
        print(f"Measure {measure.get('index', 0)}: {len(elements)} elements")
```

## 🎵 Real-World Examples

### Example 1: Processing a Scanned Piano Score

```python
import cv2
import sys
sys.path.append('src')
from omr_pipeline import OMRPipeline

# Configuration optimized for scanned piano music
config = {
    'apply_denoising': True,
    'apply_skew_correction': True,
    'confidence_threshold': 0.6,
    'staff_line_thickness': 2,
}

pipeline = OMRPipeline(config=config)

# Load the scanned image
image = cv2.imread('scanned_piano_score.png')

# Process it
result = pipeline.process_image(image=image, image_path='scanned_piano_score.png')

if result.success:
    print(f"✅ Successfully processed piano score!")
    print(f"⏱️ Took {result.total_time:.1f} seconds")
    
    # The MusicXML file is automatically generated
    print(f"📄 MusicXML saved to: {result.musicxml_path}")
    
    # You can now open this file in MuseScore, Finale, etc.
    # subprocess.run(['musescore', result.musicxml_path])  # Open in MuseScore
```

### Example 2: Processing with Manual Correction

```python
# 1. First, process automatically
result = pipeline.process_image(image=image, image_path='complex_score.png')

# 2. Check if manual correction is recommended
quality_score = len(result.detected_symbols or []) / 50  # Rough quality estimate

if quality_score < 0.7:
    print("⚠️ Low confidence - manual correction recommended")
    print("💡 Run: streamlit run src/ui/correction_interface.py")
    print("   Then upload your image for manual review")
else:
    print("✅ High confidence - automatic result looks good!")
```

### Example 3: Batch Processing a Music Library

```python
import os
from pathlib import Path
import pandas as pd

# Setup
music_library = Path("music_collection")
output_dir = Path("musicxml_output")
output_dir.mkdir(exist_ok=True)

pipeline = OMRPipeline()

# Process all images
results = []

for image_file in music_library.glob("*.png"):
    print(f"Processing: {image_file.name}")
    
    try:
        image = cv2.imread(str(image_file))
        result = pipeline.process_image(image=image, image_path=str(image_file))
        
        # Save results info
        results.append({
            'filename': image_file.name,
            'success': result.success,
            'processing_time': result.total_time,
            'symbols_detected': len(result.detected_symbols or []),
            'error': result.error_message if not result.success else None
        })
        
        print(f"   {'✅' if result.success else '❌'} {result.total_time:.1f}s")
        
    except Exception as e:
        results.append({
            'filename': image_file.name,
            'success': False,
            'processing_time': 0,
            'symbols_detected': 0,
            'error': str(e)
        })
        print(f"   ❌ Error: {e}")

# Generate report
df = pd.DataFrame(results)
df.to_csv(output_dir / "processing_report.csv", index=False)

print(f"\n📊 Batch Processing Complete!")
print(f"   ✅ Successful: {df['success'].sum()}/{len(df)}")
print(f"   ⏱️ Average time: {df['processing_time'].mean():.1f}s")
print(f"   📄 Report saved to: {output_dir}/processing_report.csv")
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Issue: Import Errors
```python
# Problem: "Module not found" errors
# Solution: Make sure you're in the right directory and add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
```

#### Issue: Low Quality Results
```python
# Problem: Poor symbol detection
# Solution: Adjust configuration
config = {
    'confidence_threshold': 0.5,      # Lower = more detections
    'apply_denoising': True,          # Clean up image
    'apply_skew_correction': True,    # Fix rotation
}
pipeline = OMRPipeline(config=config)
```

#### Issue: Memory Problems
```python
# Problem: Out of memory with large images
# Solution: Resize images before processing
import cv2

def resize_image(image, max_size=(1600, 1200)):
    h, w = image.shape[:2]
    if h > max_size[1] or w > max_size[0]:
        scale = min(max_size[0]/w, max_size[1]/h)
        new_w, new_h = int(w*scale), int(h*scale)
        return cv2.resize(image, (new_w, new_h))
    return image

# Use with pipeline
image = cv2.imread('large_score.png')
image = resize_image(image)
result = pipeline.process_image(image=image)
```

### Getting Help

1. **Check the examples**: Look in the `examples/` directory
2. **Read the docs**: Check `USAGE_GUIDE.md` for comprehensive information
3. **Test your setup**: Run `test_installation.py`
4. **Try the web interface**: Often easier for complex scores

## 🎯 Next Steps

Now that you know how to use the OMR Pipeline:

1. **Start Simple**: Try `simple_example.py` first
2. **Test Real Images**: Use your own sheet music
3. **Explore Web UI**: Try the Streamlit interface for manual correction
4. **Batch Process**: Handle multiple files at once
5. **Integrate**: Use the Python API in your own applications

### Advanced Features to Explore

- **Custom Configuration**: Tune parameters for your specific use case
- **Evaluation Metrics**: Assess pipeline performance (see `src/evaluation/`)
- **Dataset Management**: Generate synthetic training data (see `src/data/`)
- **MusicXML Integration**: Work with generated files in MuseScore, Finale, etc.

## 🎼 Happy Music Recognition!

Your OMR Pipeline is ready to transform sheet music images into digital scores. Start with the simple examples and gradually explore more advanced features as needed!

**Remember**: The system works best with clean, high-resolution images of printed music. For handwritten or low-quality images, consider using the manual correction interface for best results.