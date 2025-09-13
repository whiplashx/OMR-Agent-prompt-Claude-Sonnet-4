"""
Dataset Management Module
========================

Handles loading, augmentation, and management of sheet music datasets.
Supports IMSLP, synthetic data generation, and custom datasets.
"""

import os
import cv2
import numpy as np
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Generator
import logging
from dataclasses import dataclass
import xml.etree.ElementTree as ET

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    AUGMENTATION_AVAILABLE = True
except ImportError:
    AUGMENTATION_AVAILABLE = False
    logging.warning("Albumentations not available. Install for data augmentation.")

try:
    import requests
    from bs4 import BeautifulSoup
    WEB_SCRAPING_AVAILABLE = True
except ImportError:
    WEB_SCRAPING_AVAILABLE = False
    logging.warning("BeautifulSoup/requests not available for web scraping.")

logger = logging.getLogger(__name__)


@dataclass
class DatasetSample:
    """Represents a single dataset sample."""
    image_path: str
    image: Optional[np.ndarray] = None
    ground_truth: Optional[Dict] = None
    metadata: Optional[Dict] = None
    annotations: Optional[List] = None


class DataAugmentation:
    """
    Handles data augmentation for sheet music images.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize data augmentation.
        
        Args:
            config: Augmentation configuration
        """
        self.config = config or {}
        self.enabled = self.config.get('enabled', True) and AUGMENTATION_AVAILABLE
        
        if self.enabled:
            self.transform = self._create_augmentation_pipeline()
        else:
            self.transform = None
    
    def _create_augmentation_pipeline(self):
        """Create albumentations augmentation pipeline."""
        transforms = []
        
        # Geometric transformations
        if self.config.get('rotation', True):
            transforms.append(A.Rotate(limit=3, p=0.3))  # Small rotations
        
        if self.config.get('scale', True):
            transforms.append(A.RandomScale(scale_limit=0.1, p=0.3))
        
        if self.config.get('shear', True):
            transforms.append(A.Affine(shear=(-2, 2), p=0.2))
        
        # Photometric transformations
        if self.config.get('brightness', True):
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.3
            ))
        
        if self.config.get('noise', True):
            transforms.append(A.GaussNoise(var_limit=(10, 50), p=0.2))
        
        # Blur and sharpening
        if self.config.get('blur', True):
            transforms.append(A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.5)
            ], p=0.2))
        
        # Elastic transformations (careful with sheet music)
        if self.config.get('elastic', False):
            transforms.append(A.ElasticTransform(
                alpha=1, 
                sigma=50, 
                alpha_affine=50, 
                p=0.1
            ))
        
        return A.Compose(transforms)
    
    def augment(self, image: np.ndarray, annotations: Optional[List] = None) -> Tuple[np.ndarray, Optional[List]]:
        """
        Apply augmentation to an image and its annotations.
        
        Args:
            image: Input image
            annotations: Optional annotations (bounding boxes, keypoints)
            
        Returns:
            Augmented image and transformed annotations
        """
        if not self.enabled or self.transform is None:
            return image, annotations
        
        # Apply augmentation
        if annotations:
            # Handle bounding boxes if present
            bboxes = []
            for ann in annotations:
                if 'bbox' in ann:
                    bbox = ann['bbox']
                    # Convert to albumentations format (x_min, y_min, x_max, y_max)
                    x, y, w, h = bbox
                    bboxes.append([x, y, x + w, y + h, ann.get('class_id', 0)])
            
            if bboxes:
                transformed = self.transform(image=image, bboxes=bboxes)
                aug_image = transformed['image']
                aug_bboxes = transformed.get('bboxes', [])
                
                # Convert bboxes back to original format
                aug_annotations = []
                for i, bbox in enumerate(aug_bboxes):
                    x_min, y_min, x_max, y_max, class_id = bbox
                    ann_copy = annotations[i].copy()
                    ann_copy['bbox'] = [x_min, y_min, x_max - x_min, y_max - y_min]
                    aug_annotations.append(ann_copy)
                
                return aug_image, aug_annotations
        
        # No annotations or simple case
        transformed = self.transform(image=image)
        return transformed['image'], annotations


class SyntheticDataGenerator:
    """
    Generates synthetic sheet music images for training.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize synthetic data generator.
        
        Args:
            config: Generator configuration
        """
        self.config = config or {}
        self.image_size = self.config.get('image_size', (1200, 800))
        self.staff_spacing = self.config.get('staff_spacing', 20)
        self.line_thickness = self.config.get('line_thickness', 2)
    
    def generate_sample(self) -> Tuple[np.ndarray, Dict]:
        """
        Generate a synthetic sheet music sample.
        
        Returns:
            Generated image and ground truth annotations
        """
        height, width = self.image_size
        
        # Create blank white image
        image = np.ones((height, width), dtype=np.uint8) * 255
        
        # Generate staff lines
        staff_info = self._generate_staff_lines(image)
        
        # Generate musical symbols
        symbols = self._generate_symbols(image, staff_info)
        
        # Add noise and artifacts
        image = self._add_realistic_artifacts(image)
        
        # Create ground truth
        ground_truth = {
            'staves': staff_info,
            'symbols': symbols,
            'image_size': (width, height)
        }
        
        return image, ground_truth
    
    def _generate_staff_lines(self, image: np.ndarray) -> List[Dict]:
        """Generate staff lines on the image."""
        height, width = image.shape
        staff_info = []
        
        # Generate 2-3 staves
        num_staves = random.randint(2, 3)
        staff_height = height // num_staves
        
        for staff_idx in range(num_staves):
            staff_y_start = staff_idx * staff_height + staff_height // 4
            
            # Generate 5 staff lines
            lines = []
            for line_idx in range(5):
                y_pos = staff_y_start + line_idx * self.staff_spacing
                
                # Add some variation to line position
                y_pos += random.randint(-2, 2)
                
                # Draw the line
                start_x = random.randint(50, 100)
                end_x = width - random.randint(50, 100)
                
                cv2.line(image, (start_x, y_pos), (end_x, y_pos), 0, self.line_thickness)
                
                lines.append({
                    'y_position': y_pos,
                    'x_start': start_x,
                    'x_end': end_x,
                    'thickness': self.line_thickness
                })
            
            staff_info.append({
                'staff_index': staff_idx,
                'lines': lines,
                'y_top': staff_y_start,
                'y_bottom': staff_y_start + 4 * self.staff_spacing,
                'line_spacing': self.staff_spacing
            })
        
        return staff_info
    
    def _generate_symbols(self, image: np.ndarray, staff_info: List[Dict]) -> List[Dict]:
        """Generate musical symbols on the staves."""
        symbols = []
        
        for staff in staff_info:
            staff_y_center = (staff['y_top'] + staff['y_bottom']) // 2
            
            # Generate notes along the staff
            num_notes = random.randint(5, 15)
            x_positions = np.linspace(150, image.shape[1] - 150, num_notes)
            
            for x_pos in x_positions:
                # Random note type
                note_types = ['quarter_note', 'half_note', 'eighth_note', 'whole_note']
                note_type = random.choice(note_types)
                
                # Random vertical position (different pitches)
                y_offset = random.randint(-40, 40)
                y_pos = staff_y_center + y_offset
                
                # Draw a simple note (circle for now)
                radius = 6 if 'whole' not in note_type else 8
                thickness = -1 if note_type != 'whole_note' else 2
                
                cv2.circle(image, (int(x_pos), int(y_pos)), radius, 0, thickness)
                
                # Add stem for some notes
                if note_type in ['quarter_note', 'eighth_note', 'half_note']:
                    stem_height = 30
                    stem_y = y_pos - stem_height if y_pos > staff_y_center else y_pos + stem_height
                    cv2.line(image, (int(x_pos + radius), int(y_pos)), 
                            (int(x_pos + radius), int(stem_y)), 0, 2)
                
                # Create symbol annotation
                symbols.append({
                    'class_name': note_type,
                    'bbox': [x_pos - radius - 5, y_pos - radius - 5, 
                            2 * radius + 10, 2 * radius + 10],
                    'center': [x_pos, y_pos],
                    'confidence': 1.0,  # Perfect confidence for synthetic data
                    'staff_index': staff['staff_index']
                })
        
        return symbols
    
    def _add_realistic_artifacts(self, image: np.ndarray) -> np.ndarray:
        """Add realistic artifacts to make synthetic data more like real scans."""
        # Add noise
        noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
        image = cv2.add(image, noise)
        
        # Add slight blur
        if random.random() < 0.3:
            image = cv2.GaussianBlur(image, (3, 3), 0.5)
        
        # Add rotation
        if random.random() < 0.2:
            angle = random.uniform(-2, 2)
            center = (image.shape[1] // 2, image.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
            image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        
        return image


class DatasetManager:
    """
    Manages different types of datasets for OMR training and evaluation.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize dataset manager.
        
        Args:
            config: Dataset configuration
        """
        self.config = config or {}
        self.data_dir = Path(self.config.get('data_dir', 'data'))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.augmentation = DataAugmentation(self.config.get('augmentation', {}))
        self.synthetic_generator = SyntheticDataGenerator(self.config.get('synthetic', {}))
        
        # Dataset registry
        self.datasets = {}
    
    def register_dataset(self, name: str, dataset_path: str, dataset_type: str = 'custom'):
        """
        Register a dataset for use.
        
        Args:
            name: Dataset name
            dataset_path: Path to dataset
            dataset_type: Type of dataset ('imslp', 'custom', 'synthetic')
        """
        self.datasets[name] = {
            'path': Path(dataset_path),
            'type': dataset_type,
            'samples': []
        }
        
        logger.info(f"Registered dataset '{name}' of type '{dataset_type}'")
    
    def load_dataset(self, name: str, split: str = 'train') -> List[DatasetSample]:
        """
        Load a dataset by name.
        
        Args:
            name: Dataset name
            split: Dataset split ('train', 'val', 'test')
            
        Returns:
            List of dataset samples
        """
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not registered")
        
        dataset_info = self.datasets[name]
        dataset_type = dataset_info['type']
        
        if dataset_type == 'synthetic':
            return self._load_synthetic_dataset(split)
        elif dataset_type == 'imslp':
            return self._load_imslp_dataset(dataset_info['path'], split)
        else:
            return self._load_custom_dataset(dataset_info['path'], split)
    
    def _load_synthetic_dataset(self, split: str) -> List[DatasetSample]:
        """Load synthetic dataset samples."""
        num_samples = self.config.get('synthetic_samples', {}).get(split, 1000)
        
        samples = []
        for i in range(num_samples):
            image, ground_truth = self.synthetic_generator.generate_sample()
            
            sample = DatasetSample(
                image_path=f"synthetic_{split}_{i}",
                image=image,
                ground_truth=ground_truth,
                metadata={'synthetic': True, 'split': split}
            )
            samples.append(sample)
        
        logger.info(f"Generated {num_samples} synthetic samples for {split}")
        return samples
    
    def _load_custom_dataset(self, dataset_path: Path, split: str) -> List[DatasetSample]:
        """Load custom dataset from directory structure."""
        split_dir = dataset_path / split
        if not split_dir.exists():
            logger.warning(f"Split directory {split_dir} does not exist")
            return []
        
        samples = []
        
        # Look for images and annotations
        image_files = list(split_dir.glob('*.png')) + list(split_dir.glob('*.jpg'))
        
        for image_file in image_files:
            # Look for corresponding annotation file
            annotation_file = image_file.with_suffix('.json')
            ground_truth = None
            
            if annotation_file.exists():
                with open(annotation_file, 'r') as f:
                    ground_truth = json.load(f)
            
            sample = DatasetSample(
                image_path=str(image_file),
                ground_truth=ground_truth,
                metadata={'split': split, 'source': 'custom'}
            )
            samples.append(sample)
        
        logger.info(f"Loaded {len(samples)} samples from {split_dir}")
        return samples
    
    def _load_imslp_dataset(self, dataset_path: Path, split: str) -> List[DatasetSample]:
        """Load IMSLP dataset samples."""
        # This would implement IMSLP dataset loading
        # For now, return empty list
        logger.warning("IMSLP dataset loading not yet implemented")
        return []
    
    def create_data_loader(self, dataset_name: str, split: str, 
                          batch_size: int = 32, shuffle: bool = True) -> Generator:
        """
        Create a data loader for training.
        
        Args:
            dataset_name: Name of dataset to load
            split: Dataset split
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Yields:
            Batches of samples
        """
        samples = self.load_dataset(dataset_name, split)
        
        if shuffle:
            random.shuffle(samples)
        
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i:i + batch_size]
            
            # Load images if not already loaded
            for sample in batch_samples:
                if sample.image is None and sample.image_path:
                    if Path(sample.image_path).exists():
                        sample.image = cv2.imread(sample.image_path, cv2.IMREAD_GRAYSCALE)
            
            # Apply augmentation if enabled
            if split == 'train' and self.augmentation.enabled:
                for sample in batch_samples:
                    if sample.image is not None:
                        aug_image, aug_annotations = self.augmentation.augment(
                            sample.image, 
                            sample.ground_truth.get('symbols', []) if sample.ground_truth else None
                        )
                        sample.image = aug_image
                        if sample.ground_truth and aug_annotations:
                            sample.ground_truth['symbols'] = aug_annotations
            
            yield batch_samples
    
    def download_imslp_samples(self, num_samples: int = 100) -> None:
        """
        Download sample sheet music from IMSLP.
        
        Args:
            num_samples: Number of samples to download
        """
        if not WEB_SCRAPING_AVAILABLE:
            logger.error("Web scraping dependencies not available")
            return
        
        # This would implement IMSLP downloading
        # For now, just log a message
        logger.info(f"IMSLP downloading not implemented. Would download {num_samples} samples.")
    
    def export_dataset(self, dataset_name: str, output_dir: str, format: str = 'yolo'):
        """
        Export dataset in a specific format for training.
        
        Args:
            dataset_name: Dataset to export
            output_dir: Output directory
            format: Export format ('yolo', 'coco', 'pascal_voc')
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for split in ['train', 'val', 'test']:
            samples = self.load_dataset(dataset_name, split)
            
            if not samples:
                continue
            
            split_dir = output_path / split
            split_dir.mkdir(exist_ok=True)
            
            if format == 'yolo':
                self._export_yolo_format(samples, split_dir)
            elif format == 'coco':
                self._export_coco_format(samples, split_dir)
            else:
                logger.warning(f"Export format '{format}' not supported")
    
    def _export_yolo_format(self, samples: List[DatasetSample], output_dir: Path):
        """Export samples in YOLO format."""
        for i, sample in enumerate(samples):
            if sample.image is None:
                continue
            
            # Save image
            image_path = output_dir / f"image_{i:06d}.jpg"
            cv2.imwrite(str(image_path), sample.image)
            
            # Save annotation
            if sample.ground_truth and 'symbols' in sample.ground_truth:
                annotation_path = output_dir / f"image_{i:06d}.txt"
                
                with open(annotation_path, 'w') as f:
                    for symbol in sample.ground_truth['symbols']:
                        # Convert to YOLO format (class_id, x_center, y_center, width, height)
                        # Normalized coordinates
                        if 'bbox' in symbol:
                            bbox = symbol['bbox']
                            img_h, img_w = sample.image.shape[:2]
                            
                            x, y, w, h = bbox
                            x_center = (x + w / 2) / img_w
                            y_center = (y + h / 2) / img_h
                            width = w / img_w
                            height = h / img_h
                            
                            class_id = 0  # Simplified for now
                            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
    
    def _export_coco_format(self, samples: List[DatasetSample], output_dir: Path):
        """Export samples in COCO format."""
        # This would implement COCO format export
        logger.info("COCO format export not yet implemented")


def test_dataset_manager():
    """Test function for dataset manager."""
    # Create test dataset manager
    config = {
        'data_dir': 'test_data',
        'augmentation': {'enabled': True},
        'synthetic': {'image_size': (800, 600)},
        'synthetic_samples': {'train': 10, 'val': 5}
    }
    
    manager = DatasetManager(config)
    
    # Register synthetic dataset
    manager.register_dataset('synthetic_test', '', 'synthetic')
    
    # Load samples
    train_samples = manager.load_dataset('synthetic_test', 'train')
    print(f"Generated {len(train_samples)} training samples")
    
    # Test data loader
    batch_count = 0
    for batch in manager.create_data_loader('synthetic_test', 'train', batch_size=3):
        batch_count += 1
        print(f"Batch {batch_count}: {len(batch)} samples")
        if batch_count >= 3:  # Limit for testing
            break


if __name__ == "__main__":
    test_dataset_manager()