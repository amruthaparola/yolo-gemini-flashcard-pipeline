#!/usr/bin/env python3
"""
Flashcard Detection and Extraction using YOLOv8

Complete pipeline:
1. Setup dataset structure
2. Train YOLO model
3. Detect flashcards on new pages
4. Extract and save individual cards
"""

import os
from pathlib import Path
import shutil
from typing import List, Tuple, Optional
import argparse
import json

try:
    from ultralytics import YOLO
    from PIL import Image
    import numpy as np
    import cv2
except ImportError as e:
    print("Missing dependencies. Install with:")
    print("pip install ultralytics pillow opencv-python")
    raise e


class FlashcardDetector:
    """
    Main class for flashcard detection using YOLO.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to trained YOLO weights. If None, uses pretrained model.
        """
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
            print(f"Loaded custom model: {model_path}")
        else:
            self.model = YOLO("yolov8n.pt")  # Start with nano pretrained
            print("Using pretrained YOLOv8n model")
    
    def train(
        self,
        data_yaml: str,
        epochs: int = 30,
        imgsz: int = 1024,
        batch: int = 8,
        project: str = "runs/detect",
        name: str = "flashcard_train",
        patience: int = 10,
        **kwargs
    ):
        """
        Train the YOLO model on flashcard dataset.
        
        Args:
            data_yaml: Path to data configuration YAML
            epochs: Number of training epochs
            imgsz: Image size for training
            batch: Batch size
            project: Project directory
            name: Experiment name
            patience: Early stopping patience
            **kwargs: Additional training arguments
        """
        print("="*70)
        print("TRAINING FLASHCARD DETECTOR")
        print("="*70)
        print(f"Data config: {data_yaml}")
        print(f"Epochs: {epochs}")
        print(f"Image size: {imgsz}")
        print(f"Batch size: {batch}")
        print("="*70)
        
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=project,
            name=name,
            patience=patience,
            save=True,
            plots=True,
            **kwargs
        )
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Best model saved to: {project}/{name}/weights/best.pt")
        print(f"Results saved to: {project}/{name}/")
        
        return results
    
    def detect(
        self,
        image_path: str,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 1024
    ) -> Tuple[np.ndarray, List[Tuple[float, float, float, float]]]:
        """
        Detect flashcards in an image.
        
        Args:
            image_path: Path to input image
            conf: Confidence threshold
            iou: IoU threshold for NMS
            imgsz: Image size for inference
            
        Returns:
            Tuple of (original image as numpy array, list of bounding boxes)
            Each bbox is (x1, y1, x2, y2) in pixels
        """
        results = self.model.predict(
            image_path,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False
        )
        
        result = results[0]
        
        # Get bounding boxes
        boxes = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            
            for bbox, conf in zip(boxes_xyxy, confidences):
                x1, y1, x2, y2 = bbox
                boxes.append((float(x1), float(y1), float(x2), float(y2), float(conf)))
        
        # Load original image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Failed to load image {image_path}")
            return None, boxes
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img, boxes
    
    def extract_cards(
        self,
        image_path: str,
        output_dir: str,
        conf: float = 0.25,
        sort_by: str = "position",  # "position" or "confidence"
        padding: int = 10,
        save_visualization: bool = True
    ) -> List[str]:
        """
        Detect and extract individual flashcards from a page.
        
        Args:
            image_path: Path to input page image
            output_dir: Directory to save extracted cards
            conf: Confidence threshold
            sort_by: How to sort cards ("position" for top-to-bottom left-to-right,
                    "confidence" for highest confidence first)
            padding: Pixels to add around each card
            save_visualization: Whether to save annotated full page
            
        Returns:
            List of paths to extracted card images
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Detect cards
        img, boxes = self.detect(image_path, conf=conf)
        
        if img is None:
            print(f"Skipping {image_path}: failed to load image")
            return []
        
        if not boxes:
            print(f"No flashcards detected in {image_path}")
            return []
        
        print(f"Detected {len(boxes)} flashcards in {Path(image_path).name}")
        
        # Sort boxes
        if sort_by == "position":
            # Sort by row (y) then column (x)
            # Group by approximate row (within 50px)
            def get_row_col(box):
                x1, y1, x2, y2, conf = box
                y_center = (y1 + y2) / 2
                x_center = (x1 + x2) / 2
                row = int(y_center / 50)  # Approximate row
                return (row, x_center)
            
            boxes = sorted(boxes, key=get_row_col)
        else:  # sort by confidence
            boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
        
        # Extract and save each card
        page_name = Path(image_path).stem
        saved_paths = []
        
        for i, (x1, y1, x2, y2, confidence) in enumerate(boxes, 1):
            # Add padding
            h, w = img.shape[:2]
            x1 = max(0, int(x1) - padding)
            y1 = max(0, int(y1) - padding)
            x2 = min(w, int(x2) + padding)
            y2 = min(h, int(y2) + padding)
            
            # Extract card
            card = img[y1:y2, x1:x2]
            
            # Save
            output_path = output_dir / f"{page_name}_card_{i:02d}.png"
            card_pil = Image.fromarray(card)
            card_pil.save(output_path)
            saved_paths.append(str(output_path))
            
            print(f"  Card {i:2d}: {output_path.name:30s} "
                  f"[{x2-x1:4d}x{y2-y1:4d}] conf={confidence:.2f}")
        
        # Save visualization
        if save_visualization:
            vis_img = img.copy()
            for i, (x1, y1, x2, y2, conf) in enumerate(boxes, 1):
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw box
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Draw label
                label = f"{i}: {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(vis_img, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(vis_img, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            vis_path = output_dir / f"{page_name}_detected.png"
            vis_pil = Image.fromarray(vis_img)
            vis_pil.save(vis_path)
            print(f"  Visualization saved: {vis_path.name}")
        
        return saved_paths
    
    def batch_extract(
        self,
        input_dir: str,
        output_dir: str,
        conf: float = 0.25,
        pattern: str = "*.png",
        **kwargs
    ):
        """
        Extract flashcards from all images in a directory.
        
        Args:
            input_dir: Directory containing page images
            output_dir: Base directory for extracted cards
            conf: Confidence threshold
            pattern: File pattern to match
            **kwargs: Additional arguments for extract_cards
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        image_files = sorted(input_dir.glob(pattern))
        
        if not image_files:
            print(f"No images found matching {pattern} in {input_dir}")
            return
        
        print(f"Processing {len(image_files)} images...")
        print("="*70)
        
        total_cards = 0
        for img_path in image_files:
            try:
                page_output = output_dir / img_path.stem
                cards = self.extract_cards(str(img_path), str(page_output), conf=conf, **kwargs)
                total_cards += len(cards)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
            print()
        
        print("="*70)
        print(f"COMPLETE: Extracted {total_cards} cards from {len(image_files)} pages")
        print(f"Output directory: {output_dir}")


class DatasetSetup:
    """
    Helper class to set up YOLO dataset structure.
    """
    
    @staticmethod
    def create_structure(base_dir: str):
        """
        Create standard YOLO dataset directory structure.
        
        Args:
            base_dir: Base directory for dataset
        """
        base_dir = Path(base_dir)
        
        dirs = [
            base_dir / "images" / "train",
            base_dir / "images" / "val",
            base_dir / "labels" / "train",
            base_dir / "labels" / "val"
        ]
        
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        
        print("Created dataset structure:")
        print(f"  {base_dir}/")
        print(f"    images/")
        print(f"      train/")
        print(f"      val/")
        print(f"    labels/")
        print(f"      train/")
        print(f"      val/")
    
    @staticmethod
    def create_yaml(
        dataset_dir: str,
        output_path: str = "flashcards.yaml",
        nc: int = 1,
        names: List[str] = ["card"]
    ):
        """
        Create YOLO data configuration YAML file.
        
        Args:
            dataset_dir: Path to dataset directory
            output_path: Where to save YAML file
            nc: Number of classes
            names: List of class names
        """
        dataset_dir = Path(dataset_dir).absolute()
        
        yaml_content = f"""# Flashcard Detection Dataset Configuration

path: {dataset_dir}  # dataset root
train: images/train  # train images (relative to path)
val: images/val      # validation images (relative to path)

# Classes
nc: {nc}             # number of classes
names: {names}       # class names
"""
        
        with open(output_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"Created data config: {output_path}")
        print(f"  Path: {dataset_dir}")
        print(f"  Classes: {names}")
    
    @staticmethod
    def split_dataset(
        images_dir: str,
        labels_dir: str,
        output_dir: str,
        val_split: float = 0.2,
        seed: int = 42
    ):
        """
        Split labeled data into train/val sets.
        
        Args:
            images_dir: Directory containing all images
            labels_dir: Directory containing all labels
            output_dir: Output dataset directory
            val_split: Fraction for validation
            seed: Random seed
        """
        import random
        random.seed(seed)
        
        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)
        output_dir = Path(output_dir)
        
        # Get all images with corresponding labels
        image_files = []
        for img_path in sorted(images_dir.glob("*.png")):
            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                image_files.append(img_path)
        
        if not image_files:
            print("No matching image-label pairs found!")
            return
        
        # Shuffle and split
        random.shuffle(image_files)
        split_idx = int(len(image_files) * (1 - val_split))
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        print(f"Splitting {len(image_files)} files:")
        print(f"  Train: {len(train_files)}")
        print(f"  Val:   {len(val_files)}")
        
        # Create structure
        DatasetSetup.create_structure(output_dir)
        
        # Copy files
        for split_name, file_list in [("train", train_files), ("val", val_files)]:
            for img_path in file_list:
                label_path = labels_dir / f"{img_path.stem}.txt"
                
                # Copy image
                shutil.copy(img_path, output_dir / "images" / split_name / img_path.name)
                
                # Copy label
                shutil.copy(label_path, output_dir / "labels" / split_name / label_path.name)
        
        print(f"Dataset ready at: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Flashcard Detection using YOLOv8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup dataset structure
  python flashcard_yolo.py setup --dataset-dir ./dataset
  
  # Create data YAML
  python flashcard_yolo.py create-yaml --dataset-dir ./dataset
  
  # Train model
  python flashcard_yolo.py train --data ./flashcards.yaml --epochs 30
  
  # Extract cards from a page
  python flashcard_yolo.py extract --model ./best.pt --input page.png --output ./cards
  
  # Batch extract from directory
  python flashcard_yolo.py batch --model ./best.pt --input ./pages --output ./all_cards
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Create dataset structure")
    setup_parser.add_argument("--dataset-dir", required=True, help="Dataset directory")
    
    # Create YAML command
    yaml_parser = subparsers.add_parser("create-yaml", help="Create data YAML")
    yaml_parser.add_argument("--dataset-dir", required=True, help="Dataset directory")
    yaml_parser.add_argument("--output", default="flashcards.yaml", help="Output YAML file")
    
    # Split dataset command
    split_parser = subparsers.add_parser("split", help="Split labeled data into train/val")
    split_parser.add_argument("--images", required=True, help="Directory with images")
    split_parser.add_argument("--labels", required=True, help="Directory with labels")
    split_parser.add_argument("--output", required=True, help="Output dataset directory")
    split_parser.add_argument("--val-split", type=float, default=0.2, help="Validation split")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train YOLO model")
    train_parser.add_argument("--data", required=True, help="Path to data YAML")
    train_parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    train_parser.add_argument("--imgsz", type=int, default=1024, help="Image size")
    train_parser.add_argument("--batch", type=int, default=8, help="Batch size")
    train_parser.add_argument("--name", default="flashcard_train", help="Experiment name")
    
    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract cards from single page")
    extract_parser.add_argument("--model", required=True, help="Path to trained model")
    extract_parser.add_argument("--input", required=True, help="Input page image")
    extract_parser.add_argument("--output", required=True, help="Output directory")
    extract_parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    
    # Batch extract command
    batch_parser = subparsers.add_parser("batch", help="Extract cards from multiple pages")
    batch_parser.add_argument("--model", required=True, help="Path to trained model")
    batch_parser.add_argument("--input", required=True, help="Input directory")
    batch_parser.add_argument("--output", required=True, help="Output directory")
    batch_parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    batch_parser.add_argument("--pattern", default="*.png", help="File pattern")
    
    args = parser.parse_args()
    
    if args.command == "setup":
        DatasetSetup.create_structure(args.dataset_dir)
    
    elif args.command == "create-yaml":
        DatasetSetup.create_yaml(args.dataset_dir, args.output)
    
    elif args.command == "split":
        DatasetSetup.split_dataset(args.images, args.labels, args.output, args.val_split)
    
    elif args.command == "train":
        detector = FlashcardDetector()
        detector.train(
            data_yaml=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            name=args.name
        )
    
    elif args.command == "extract":
        detector = FlashcardDetector(args.model)
        detector.extract_cards(args.input, args.output, conf=args.conf)
    
    elif args.command == "batch":
        detector = FlashcardDetector(args.model)
        detector.batch_extract(
            args.input, args.output, conf=args.conf, pattern=args.pattern
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()