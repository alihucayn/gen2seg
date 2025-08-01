# 📊 Document Forgery Detection Evaluation

This module provides comprehensive evaluation metrics for document forgery detection models, including IoU (Intersection over Union), F-score, Precision, and Recall.

## 🎯 Features

### **Comprehensive Metrics**
- **IoU (Intersection over Union)**: Measures overlap between predicted and ground truth regions
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: Ratio of correctly predicted forgery pixels to all predicted forgery pixels
- **Recall**: Ratio of correctly predicted forgery pixels to all actual forgery pixels
- **Accuracy**: Overall pixel accuracy
- **Dice Coefficient**: Alternative overlap measure

### **Flexible Input Formats**
- **NumPy Arrays**: Direct array evaluation
- **PyTorch Tensors**: Automatic conversion and evaluation
- **Batch Processing**: Evaluate multiple images at once
- **Single Images**: Evaluate individual predictions

### **Configurable Parameters**
- **Threshold**: Binary classification threshold (0-1)
- **Ignore Index**: Pixels to ignore during evaluation
- **Multi-class Support**: Extensible for multiple classes

## 🚀 Quick Start

### **Basic Usage**

```python
from eval.metrics.iou_metric import DocumentForgeryEvaluator

# Create evaluator
evaluator = DocumentForgeryEvaluator(threshold=0.5)

# Evaluate single prediction
metrics = evaluator.evaluate_batch(prediction, target)
evaluator.print_metrics(metrics)
```

### **Batch Evaluation**

```python
# Evaluate multiple images
predictions = [pred1, pred2, pred3]  # List of numpy arrays
targets = [target1, target2, target3]  # List of numpy arrays

metrics = evaluator.evaluate_dataset(predictions, targets)
```

### **Command Line Evaluation**

```bash
# Evaluate model predictions
python eval/evaluate_model.py \
    --pred_dir /path/to/predictions \
    --target_dir /path/to/ground_truth \
    --threshold 0.5 \
    --output_file results.json
```

## 📋 Input Requirements

### **Image Format**
- **Shape**: `[H, W]` or `[B, H, W]` for batches
- **Values**: 
  - `0` (black) = authentic regions
  - `255` (white) = forgery regions
  - Or continuous values `[0, 255]` (will be thresholded)

### **Data Types**
- **NumPy Arrays**: `np.ndarray`
- **PyTorch Tensors**: `torch.Tensor`
- **File Formats**: PNG, JPG (loaded as grayscale)

## 🔧 Configuration

### **Threshold Selection**
```python
# Conservative threshold (fewer false positives)
evaluator = DocumentForgeryEvaluator(threshold=0.7)

# Balanced threshold
evaluator = DocumentForgeryEvaluator(threshold=0.5)

# Liberal threshold (fewer false negatives)
evaluator = DocumentForgeryEvaluator(threshold=0.3)
```

### **Ignore Regions**
```python
# Ignore certain regions during evaluation
evaluator = DocumentForgeryEvaluator(ignore_index=128)
```

## 📊 Output Metrics

### **Primary Metrics**
- **IoU**: `0.0` to `1.0` (higher is better)
- **F1-Score**: `0.0` to `1.0` (higher is better)
- **Precision**: `0.0` to `1.0` (higher is better)
- **Recall**: `0.0` to `1.0` (higher is better)

### **Additional Metrics**
- **Accuracy**: Overall pixel accuracy
- **Dice**: Alternative overlap measure
- **Confusion Matrix**: TP, FP, FN, TN counts

### **Example Output**
```
==================================================
Document Forgery Detection Evaluation
==================================================
Precision:  0.8542
Recall:     0.9234
F1-Score:   0.8873
IoU:        0.7956
Dice:       0.8873
Accuracy:   0.9456

Confusion Matrix:
True Positives:  12345
False Positives: 2345
False Negatives: 1234
True Negatives:  45678
==================================================
```

## 🧪 Testing

### **Run Test Suite**
```bash
python eval/test_iou_metric.py
```

### **Test Scenarios**
1. **Perfect Predictions**: Should give 1.0 for all metrics
2. **Random Predictions**: Baseline performance
3. **Batch Evaluation**: Multiple images
4. **Different Thresholds**: Impact of threshold selection
5. **PyTorch Tensors**: Tensor compatibility
6. **Edge Cases**: Empty targets, all forgery

## 📁 File Structure

```
eval/
├── metrics/
│   └── iou_metric.py          # Main IoU metric implementation
├── evaluate_model.py           # Command-line evaluation script
├── test_iou_metric.py         # Test suite
└── README.md                  # This file
```

## 🔍 Advanced Usage

### **Custom Evaluation Pipeline**

```python
from eval.metrics.iou_metric import DocumentForgeryIoUMetric

# Create metric instance
metric = DocumentForgeryIoUMetric(threshold=0.5)

# Update with predictions
for pred, target in zip(predictions, targets):
    metric.update(pred, target)

# Compute final metrics
results = metric.compute()
print(f"IoU: {results['iou']:.4f}")
```

### **Multi-class Evaluation**

```python
# For multi-class segmentation
evaluator = DocumentForgeryEvaluator(
    threshold=0.5,
    num_classes=3,  # Background, authentic, forgery
    average='macro'
)
```

### **Ignore Masks**

```python
# Use ignore masks for evaluation
evaluator.evaluate_batch(prediction, target, mask=ignore_mask)
```

## 🎯 Best Practices

### **Threshold Selection**
1. **Start with 0.5**: Balanced precision/recall
2. **High precision needed**: Use 0.7-0.9
3. **High recall needed**: Use 0.3-0.5
4. **Validate on test set**: Choose optimal threshold

### **Evaluation Protocol**
1. **Use test set**: Never evaluate on training data
2. **Consistent preprocessing**: Same normalization as training
3. **Multiple thresholds**: Report performance across thresholds
4. **Statistical significance**: Use proper statistical tests

### **Interpretation**
- **IoU > 0.8**: Excellent performance
- **IoU > 0.6**: Good performance
- **IoU > 0.4**: Acceptable performance
- **IoU < 0.4**: Needs improvement

## 🐛 Troubleshooting

### **Common Issues**

**Zero IoU**: Check if predictions and targets have same scale
```python
# Ensure same value range
prediction = prediction.astype(np.uint8)
target = target.astype(np.uint8)
```

**Shape Mismatch**: Ensure same dimensions
```python
assert prediction.shape == target.shape, "Shape mismatch"
```

**Memory Issues**: Use batch processing for large datasets
```python
# Process in batches
for batch_pred, batch_target in dataloader:
    evaluator.evaluate_batch(batch_pred, batch_target)
```

## 📈 Performance Tips

1. **Use NumPy**: Faster than PyTorch for evaluation
2. **Batch Processing**: More efficient than single images
3. **Memory Management**: Clear variables after evaluation
4. **Parallel Processing**: Use multiple processes for large datasets

## 🤝 Contributing

To extend the evaluation metrics:

1. **Add new metrics**: Extend `DocumentForgeryIoUMetric`
2. **Support new formats**: Add input validation
3. **Improve performance**: Optimize computation
4. **Add tests**: Ensure reliability

## 📚 References

- **IoU (Jaccard Index)**: Standard segmentation metric
- **F1-Score**: Harmonic mean of precision and recall
- **Dice Coefficient**: Alternative overlap measure
- **Confusion Matrix**: Fundamental classification metrics 