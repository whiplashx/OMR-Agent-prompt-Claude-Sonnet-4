# OMR Evaluation System

This directory contains comprehensive evaluation tools for the OMR pipeline.

## Components

### metrics.py
Core evaluation metrics including:
- **Symbol-level evaluation**: Detection precision/recall, classification accuracy, confidence calibration
- **Semantic-level evaluation**: Pitch accuracy, rhythm accuracy, key/time signature detection
- **Overall musical similarity**: Composite scoring system

### evaluation_cli.py
Command-line interface for running evaluations on datasets:
```bash
python evaluation_cli.py dataset_path output_dir --config config.json
```

## Usage

### Basic Evaluation
```python
from src.evaluation import OMREvaluator

evaluator = OMREvaluator()
results = evaluator.evaluate_full_pipeline(predicted_results, ground_truth)
report = evaluator.generate_evaluation_report(results)
```

### Dataset Evaluation
```python
from src.evaluation import EvaluationRunner

runner = EvaluationRunner('config.json')
results = runner.evaluate_dataset('path/to/dataset', 'output/dir')
```

## Metrics

### Symbol-Level Metrics
- **Detection**: Precision, Recall, F1-score using IoU matching
- **Classification**: Accuracy among detected symbols
- **Per-class**: Individual performance for each symbol type
- **Confidence Calibration**: Expected Calibration Error (ECE)

### Semantic-Level Metrics
- **Pitch Accuracy**: Correctness of detected pitches
- **Rhythm Accuracy**: Correctness of note durations
- **Key Signature**: Detection accuracy of key signatures
- **Time Signature**: Detection accuracy of time signatures
- **Overall Musical Similarity**: Weighted composite score

## Configuration

Example configuration file:
```json
{
  "evaluation": {
    "iou_threshold": 0.5,
    "confidence_threshold": 0.3
  }
}
```

## Ground Truth Format

Expected ground truth format for evaluation:
```json
{
  "symbols": [
    {
      "class_name": "quarter_note",
      "bbox": [x, y, width, height],
      "pitch": "C4",
      "confidence": 1.0
    }
  ],
  "musical_elements": {
    "staves": [
      {
        "elements": [...],
        "key_signature": {"sharps": 0, "key": "C"},
        "time_signature": {"numerator": 4, "denominator": 4}
      }
    ]
  }
}
```