#!/usr/bin/env python3
"""
Evaluation script for document forgery detection models.

This script demonstrates how to use the IoU metric to evaluate
model predictions against ground truth labels.
"""

import os
import argparse
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional

# Import the IoU metric
from eval.metrics.iou_metric import DocumentForgeryEvaluator, compute_iou_metrics

def load_image_as_array(image_path: str) -> np.ndarray:
    """
    Load an image and convert to numpy array.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Numpy array of the image
    """
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    return np.array(image)

def load_predictions_and_targets(pred_dir: str, 
                                target_dir: str,
                                file_list: Optional[List[str]] = None) -> tuple:
    """
    Load predictions and targets from directories.
    
    Args:
        pred_dir: Directory containing prediction images
        target_dir: Directory containing ground truth images
        file_list: Optional list of specific files to evaluate
    
    Returns:
        Tuple of (predictions, targets) as lists of numpy arrays
    """
    pred_path = Path(pred_dir)
    target_path = Path(target_dir)
    
    predictions = []
    targets = []
    
    if file_list is None:
        # Get all files from prediction directory
        pred_files = list(pred_path.glob('*.png')) + list(pred_path.glob('*.jpg'))
    else:
        # Use specific file list
        pred_files = [pred_path / f for f in file_list]
    
    for pred_file in pred_files:
        # Construct target file path
        target_file = target_path / pred_file.name
        
        if not target_file.exists():
            print(f"Warning: Target file {target_file} not found, skipping {pred_file}")
            continue
        
        # Load images
        pred_array = load_image_as_array(str(pred_file))
        target_array = load_image_as_array(str(target_file))
        
        # Ensure same shape
        if pred_array.shape != target_array.shape:
            print(f"Warning: Shape mismatch for {pred_file.name}, skipping")
            continue
        
        predictions.append(pred_array)
        targets.append(target_array)
    
    return predictions, targets

def evaluate_model_predictions(pred_dir: str,
                             target_dir: str,
                             threshold: float = 0.5,
                             ignore_index: int = 255,
                             output_file: Optional[str] = None) -> Dict[str, float]:
    """
    Evaluate model predictions using IoU metrics.
    
    Args:
        pred_dir: Directory containing prediction images
        target_dir: Directory containing ground truth images
        threshold: Threshold for binary classification
        ignore_index: Index to ignore in evaluation
        output_file: Optional file to save results
    
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"Loading predictions from: {pred_dir}")
    print(f"Loading targets from: {target_dir}")
    
    # Load predictions and targets
    predictions, targets = load_predictions_and_targets(pred_dir, target_dir)
    
    if not predictions:
        raise ValueError("No valid prediction-target pairs found!")
    
    print(f"Evaluating {len(predictions)} images...")
    
    # Create evaluator
    evaluator = DocumentForgeryEvaluator(
        threshold=threshold,
        ignore_index=ignore_index
    )
    
    # Evaluate dataset
    metrics = evaluator.evaluate_dataset(predictions, targets)
    
    # Print results
    evaluator.print_metrics(metrics, "Document Forgery Detection Evaluation")
    
    # Save results if requested
    if output_file:
        import json
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Results saved to: {output_file}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate document forgery detection model")
    
    parser.add_argument("--pred_dir", type=str, required=True,
                       help="Directory containing prediction images")
    parser.add_argument("--target_dir", type=str, required=True,
                       help="Directory containing ground truth images")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Threshold for binary classification (0-1)")
    parser.add_argument("--ignore_index", type=int, default=255,
                       help="Index to ignore in evaluation")
    parser.add_argument("--output_file", type=str, default=None,
                       help="File to save evaluation results")
    
    args = parser.parse_args()
    
    # Validate directories
    if not os.path.exists(args.pred_dir):
        raise ValueError(f"Prediction directory does not exist: {args.pred_dir}")
    if not os.path.exists(args.target_dir):
        raise ValueError(f"Target directory does not exist: {args.target_dir}")
    
    # Run evaluation
    metrics = evaluate_model_predictions(
        pred_dir=args.pred_dir,
        target_dir=args.target_dir,
        threshold=args.threshold,
        ignore_index=args.ignore_index,
        output_file=args.output_file
    )
    
    return metrics

if __name__ == "__main__":
    main() 