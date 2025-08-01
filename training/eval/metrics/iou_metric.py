#!/usr/bin/env python3
"""
IoU Metric for Document Forgery Detection

This module provides comprehensive evaluation metrics for document forgery detection,
including IoU (Intersection over Union), F-score, Precision, and Recall.

The metric handles binary segmentation masks where:
- 0 (black) = authentic regions
- 255 (white) = forgery regions
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class DocumentForgeryIoUMetric:
    """
    Comprehensive IoU metric for document forgery detection evaluation.
    
    This metric calculates:
    - IoU (Intersection over Union)
    - F-score (F1-score)
    - Precision
    - Recall
    - Accuracy
    - Dice coefficient
    
    Supports both binary and multi-class evaluation.
    """
    
    def __init__(self, 
                 threshold: float = 0.5,
                 ignore_index: int = 255,
                 num_classes: int = 2,
                 average: str = 'binary',
                 eps: float = 1e-8):
        """
        Initialize the IoU metric.
        
        Args:
            threshold (float): Threshold for binary classification (0-1)
            ignore_index (int): Index to ignore in evaluation
            num_classes (int): Number of classes (2 for binary)
            average (str): Averaging method ('binary', 'micro', 'macro', 'weighted')
            eps (float): Small epsilon to avoid division by zero
        """
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.average = average
        self.eps = eps
        
        # Storage for accumulating metrics
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.tp = 0  # True Positives
        self.fp = 0  # False Positives
        self.fn = 0  # False Negatives
        self.tn = 0  # True Negatives
        self.total_pixels = 0
        
        # Per-class metrics
        self.class_tp = defaultdict(int)
        self.class_fp = defaultdict(int)
        self.class_fn = defaultdict(int)
        self.class_tn = defaultdict(int)
    
    def update(self, 
               pred: Union[np.ndarray, torch.Tensor],
               target: Union[np.ndarray, torch.Tensor],
               mask: Optional[Union[np.ndarray, torch.Tensor]] = None):
        """
        Update metrics with new predictions and targets.
        
        Args:
            pred: Predicted segmentation mask [H, W] or [B, H, W]
            target: Ground truth segmentation mask [H, W] or [B, H, W]
            mask: Optional mask to ignore certain regions [H, W] or [B, H, W]
        """
        # Convert to numpy if needed
        if torch.is_tensor(pred):
            pred = pred.detach().cpu().numpy()
        if torch.is_tensor(target):
            target = target.detach().cpu().numpy()
        if torch.is_tensor(mask):
            mask = mask.detach().cpu().numpy()
        
        # Handle batch dimension
        if pred.ndim == 3:
            for i in range(pred.shape[0]):
                self._update_single(pred[i], target[i], mask[i] if mask is not None else None)
        else:
            self._update_single(pred, target, mask)
    
    def _update_single(self, 
                       pred: np.ndarray,
                       target: np.ndarray,
                       mask: Optional[np.ndarray] = None):
        """
        Update metrics for a single image.
        
        Args:
            pred: Predicted mask [H, W]
            target: Ground truth mask [H, W]
            mask: Optional ignore mask [H, W]
        """
        # Normalize predictions to [0, 1] range
        if pred.max() > 1:
            pred = pred.astype(np.float32) / 255.0
        
        # Apply threshold for binary classification
        pred_binary = (pred > self.threshold).astype(np.uint8)
        target_binary = (target > self.threshold).astype(np.uint8)
        
        # Apply ignore mask if provided
        if mask is not None:
            ignore_mask = (mask == self.ignore_index)
            pred_binary[ignore_mask] = 0
            target_binary[ignore_mask] = 0
        
        # Calculate confusion matrix
        tp = np.sum((pred_binary == 1) & (target_binary == 1))
        fp = np.sum((pred_binary == 1) & (target_binary == 0))
        fn = np.sum((pred_binary == 0) & (target_binary == 1))
        tn = np.sum((pred_binary == 0) & (target_binary == 0))
        
        # Accumulate metrics
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn
        self.total_pixels += pred_binary.size
        
        # Per-class metrics (for multi-class if needed)
        for class_id in range(self.num_classes):
            class_pred = (pred_binary == class_id)
            class_target = (target_binary == class_id)
            
            self.class_tp[class_id] += np.sum(class_pred & class_target)
            self.class_fp[class_id] += np.sum(class_pred & ~class_target)
            self.class_fn[class_id] += np.sum(~class_pred & class_target)
            self.class_tn[class_id] += np.sum(~class_pred & ~class_target)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.
        
        Returns:
            Dictionary containing all computed metrics
        """
        metrics = {}
        
        # Binary metrics
        if self.average == 'binary':
            metrics.update(self._compute_binary_metrics())
        else:
            # Multi-class metrics
            metrics.update(self._compute_multiclass_metrics())
        
        return metrics
    
    def _compute_binary_metrics(self) -> Dict[str, float]:
        """Compute binary classification metrics."""
        # Avoid division by zero
        if self.tp + self.fp == 0:
            precision = 0.0
        else:
            precision = self.tp / (self.tp + self.fp + self.eps)
        
        if self.tp + self.fn == 0:
            recall = 0.0
        else:
            recall = self.tp / (self.tp + self.fn + self.eps)
        
        # F1-score
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall + self.eps)
        
        # IoU (Jaccard Index)
        if self.tp + self.fp + self.fn == 0:
            iou = 0.0
        else:
            iou = self.tp / (self.tp + self.fp + self.fn + self.eps)
        
        # Dice coefficient
        if 2 * self.tp + self.fp + self.fn == 0:
            dice = 0.0
        else:
            dice = 2 * self.tp / (2 * self.tp + self.fp + self.fn + self.eps)
        
        # Accuracy
        if self.total_pixels == 0:
            accuracy = 0.0
        else:
            accuracy = (self.tp + self.tn) / (self.total_pixels + self.eps)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'iou': iou,
            'dice': dice,
            'accuracy': accuracy,
            'tp': self.tp,
            'fp': self.fp,
            'fn': self.fn,
            'tn': self.tn
        }
    
    def _compute_multiclass_metrics(self) -> Dict[str, float]:
        """Compute multi-class metrics."""
        metrics = {}
        
        # Per-class metrics
        for class_id in range(self.num_classes):
            tp = self.class_tp[class_id]
            fp = self.class_fp[class_id]
            fn = self.class_fn[class_id]
            tn = self.class_tn[class_id]
            
            # Precision
            if tp + fp == 0:
                precision = 0.0
            else:
                precision = tp / (tp + fp + self.eps)
            
            # Recall
            if tp + fn == 0:
                recall = 0.0
            else:
                recall = tp / (tp + fn + self.eps)
            
            # F1-score
            if precision + recall == 0:
                f1_score = 0.0
            else:
                f1_score = 2 * (precision * recall) / (precision + recall + self.eps)
            
            # IoU
            if tp + fp + fn == 0:
                iou = 0.0
            else:
                iou = tp / (tp + fp + fn + self.eps)
            
            metrics[f'class_{class_id}_precision'] = precision
            metrics[f'class_{class_id}_recall'] = recall
            metrics[f'class_{class_id}_f1_score'] = f1_score
            metrics[f'class_{class_id}_iou'] = iou
        
        # Average metrics
        if self.average == 'macro':
            metrics['precision'] = np.mean([metrics[f'class_{i}_precision'] for i in range(self.num_classes)])
            metrics['recall'] = np.mean([metrics[f'class_{i}_recall'] for i in range(self.num_classes)])
            metrics['f1_score'] = np.mean([metrics[f'class_{i}_f1_score'] for i in range(self.num_classes)])
            metrics['iou'] = np.mean([metrics[f'class_{i}_iou'] for i in range(self.num_classes)])
        
        return metrics


class DocumentForgeryEvaluator:
    """
    High-level evaluator for document forgery detection models.
    
    This class provides easy-to-use evaluation methods for comparing
    predicted segmentation masks against ground truth.
    """
    
    def __init__(self, 
                 threshold: float = 0.5,
                 ignore_index: int = 255,
                 num_classes: int = 2):
        """
        Initialize the evaluator.
        
        Args:
            threshold (float): Threshold for binary classification
            ignore_index (int): Index to ignore in evaluation
            num_classes (int): Number of classes
        """
        self.metric = DocumentForgeryIoUMetric(
            threshold=threshold,
            ignore_index=ignore_index,
            num_classes=num_classes
        )
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.num_classes = num_classes
    
    def evaluate_batch(self, 
                      predictions: Union[np.ndarray, torch.Tensor],
                      targets: Union[np.ndarray, torch.Tensor],
                      masks: Optional[Union[np.ndarray, torch.Tensor]] = None) -> Dict[str, float]:
        """
        Evaluate a batch of predictions.
        
        Args:
            predictions: Predicted masks [B, H, W] or [B, C, H, W]
            targets: Ground truth masks [B, H, W] or [B, C, H, W]
            masks: Optional ignore masks [B, H, W]
        
        Returns:
            Dictionary of computed metrics
        """
        self.metric.reset()
        self.metric.update(predictions, targets, masks)
        return self.metric.compute()
    
    def evaluate_dataset(self, 
                        predictions: List[Union[np.ndarray, torch.Tensor]],
                        targets: List[Union[np.ndarray, torch.Tensor]],
                        masks: Optional[List[Union[np.ndarray, torch.Tensor]]] = None) -> Dict[str, float]:
        """
        Evaluate a full dataset.
        
        Args:
            predictions: List of predicted masks
            targets: List of ground truth masks
            masks: Optional list of ignore masks
        
        Returns:
            Dictionary of computed metrics
        """
        self.metric.reset()
        
        for i, (pred, target) in enumerate(zip(predictions, targets)):
            mask = masks[i] if masks is not None else None
            self.metric.update(pred, target, mask)
        
        return self.metric.compute()
    
    def print_metrics(self, metrics: Dict[str, float], title: str = "Evaluation Results"):
        """
        Print metrics in a formatted way.
        
        Args:
            metrics: Dictionary of metrics
            title: Title for the output
        """
        print(f"\n{'='*50}")
        print(f"{title}")
        print(f"{'='*50}")
        
        # Main metrics
        if 'precision' in metrics:
            print(f"Precision:  {metrics['precision']:.4f}")
        if 'recall' in metrics:
            print(f"Recall:     {metrics['recall']:.4f}")
        if 'f1_score' in metrics:
            print(f"F1-Score:   {metrics['f1_score']:.4f}")
        if 'iou' in metrics:
            print(f"IoU:        {metrics['iou']:.4f}")
        if 'dice' in metrics:
            print(f"Dice:       {metrics['dice']:.4f}")
        if 'accuracy' in metrics:
            print(f"Accuracy:   {metrics['accuracy']:.4f}")
        
        # Confusion matrix counts
        if 'tp' in metrics:
            print(f"\nConfusion Matrix:")
            print(f"True Positives:  {metrics['tp']}")
            print(f"False Positives: {metrics['fp']}")
            print(f"False Negatives: {metrics['fn']}")
            print(f"True Negatives:  {metrics['tn']}")
        
        print(f"{'='*50}\n")


def compute_iou_metrics(predictions: Union[np.ndarray, torch.Tensor],
                       targets: Union[np.ndarray, torch.Tensor],
                       threshold: float = 0.5,
                       ignore_index: int = 255) -> Dict[str, float]:
    """
    Quick function to compute IoU metrics.
    
    Args:
        predictions: Predicted masks
        targets: Ground truth masks
        threshold: Threshold for binary classification
        ignore_index: Index to ignore
    
    Returns:
        Dictionary of metrics
    """
    evaluator = DocumentForgeryEvaluator(threshold=threshold, ignore_index=ignore_index)
    return evaluator.evaluate_batch(predictions, targets)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    import torch
    
    # Create sample data
    batch_size = 4
    height, width = 512, 512
    
    # Random predictions and targets
    predictions = torch.rand(batch_size, height, width) * 255
    targets = torch.randint(0, 256, (batch_size, height, width))
    
    # Create evaluator
    evaluator = DocumentForgeryEvaluator(threshold=0.5)
    
    # Evaluate
    metrics = evaluator.evaluate_batch(predictions, targets)
    
    # Print results
    evaluator.print_metrics(metrics, "Sample Evaluation")
    
    print("Individual metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
