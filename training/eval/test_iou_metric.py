#!/usr/bin/env python3
"""
Test script for IoU metric functionality.

This script demonstrates various use cases of the IoU metric
for document forgery detection evaluation.
"""

import numpy as np
import torch
from eval.metrics.iou_metric import DocumentForgeryEvaluator, compute_iou_metrics

def test_perfect_predictions():
    """Test with perfect predictions (should give 1.0 for all metrics)."""
    print("\n" + "="*60)
    print("TEST 1: Perfect Predictions")
    print("="*60)
    
    # Create perfect predictions (same as targets)
    height, width = 512, 512
    target = np.zeros((height, width), dtype=np.uint8)
    
    # Add some forgery regions (white areas)
    target[100:200, 100:300] = 255
    target[300:400, 200:400] = 255
    
    # Perfect prediction (same as target)
    prediction = target.copy()
    
    # Evaluate
    evaluator = DocumentForgeryEvaluator(threshold=0.5)
    metrics = evaluator.evaluate_batch(prediction, target)
    evaluator.print_metrics(metrics, "Perfect Predictions")
    
    return metrics

def test_random_predictions():
    """Test with random predictions."""
    print("\n" + "="*60)
    print("TEST 2: Random Predictions")
    print("="*60)
    
    height, width = 512, 512
    
    # Create target with some forgery regions
    target = np.zeros((height, width), dtype=np.uint8)
    target[100:200, 100:300] = 255
    target[300:400, 200:400] = 255
    
    # Random predictions
    prediction = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    
    # Evaluate
    evaluator = DocumentForgeryEvaluator(threshold=0.5)
    metrics = evaluator.evaluate_batch(prediction, target)
    evaluator.print_metrics(metrics, "Random Predictions")
    
    return metrics

def test_batch_evaluation():
    """Test batch evaluation with multiple images."""
    print("\n" + "="*60)
    print("TEST 3: Batch Evaluation")
    print("="*60)
    
    batch_size = 4
    height, width = 256, 256
    
    # Create batch of targets
    targets = []
    predictions = []
    
    for i in range(batch_size):
        # Create target with forgery regions
        target = np.zeros((height, width), dtype=np.uint8)
        target[50:100, 50:150] = 255
        target[150:200, 100:200] = 255
        
        # Create prediction (somewhat similar to target)
        prediction = target.copy()
        # Add some noise
        noise = np.random.randint(0, 100, (height, width), dtype=np.uint8)
        prediction = np.clip(prediction + noise, 0, 255)
        
        targets.append(target)
        predictions.append(prediction)
    
    # Evaluate batch
    evaluator = DocumentForgeryEvaluator(threshold=0.5)
    metrics = evaluator.evaluate_dataset(predictions, targets)
    evaluator.print_metrics(metrics, "Batch Evaluation")
    
    return metrics

def test_different_thresholds():
    """Test how different thresholds affect the metrics."""
    print("\n" + "="*60)
    print("TEST 4: Different Thresholds")
    print("="*60)
    
    height, width = 256, 256
    
    # Create target
    target = np.zeros((height, width), dtype=np.uint8)
    target[50:150, 50:200] = 255
    
    # Create prediction with continuous values
    prediction = np.random.rand(height, width) * 255
    
    # Test different thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for threshold in thresholds:
        evaluator = DocumentForgeryEvaluator(threshold=threshold)
        metrics = evaluator.evaluate_batch(prediction, target)
        
        print(f"\nThreshold: {threshold}")
        print(f"IoU: {metrics['iou']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")

def test_torch_tensors():
    """Test with PyTorch tensors."""
    print("\n" + "="*60)
    print("TEST 5: PyTorch Tensors")
    print("="*60)
    
    height, width = 256, 256
    
    # Create target and prediction as tensors
    target = torch.zeros(height, width, dtype=torch.uint8)
    target[50:150, 50:200] = 255
    
    prediction = torch.rand(height, width) * 255
    
    # Evaluate
    metrics = compute_iou_metrics(prediction, target, threshold=0.5)
    
    evaluator = DocumentForgeryEvaluator(threshold=0.5)
    evaluator.print_metrics(metrics, "PyTorch Tensors")
    
    return metrics

def test_edge_cases():
    """Test edge cases like empty predictions/targets."""
    print("\n" + "="*60)
    print("TEST 6: Edge Cases")
    print("="*60)
    
    height, width = 256, 256
    
    # Case 1: Empty target (no forgery)
    target_empty = np.zeros((height, width), dtype=np.uint8)
    prediction_random = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    
    evaluator = DocumentForgeryEvaluator(threshold=0.5)
    metrics = evaluator.evaluate_batch(prediction_random, target_empty)
    evaluator.print_metrics(metrics, "Empty Target (No Forgery)")
    
    # Case 2: All forgery target
    target_all = np.full((height, width), 255, dtype=np.uint8)
    prediction_empty = np.zeros((height, width), dtype=np.uint8)
    
    metrics = evaluator.evaluate_batch(prediction_empty, target_all)
    evaluator.print_metrics(metrics, "All Forgery Target")

def main():
    """Run all tests."""
    print("IoU Metric Test Suite")
    print("="*60)
    
    # Run all tests
    test_perfect_predictions()
    test_random_predictions()
    test_batch_evaluation()
    test_different_thresholds()
    test_torch_tensors()
    test_edge_cases()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)

if __name__ == "__main__":
    main() 