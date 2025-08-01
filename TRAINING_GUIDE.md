# 🚀 Document Forgery Detection Training Guide

This guide will help you train a Stable Diffusion model for document forgery detection using the gen2seg framework.

## 📋 Prerequisites

### 1. Install Dependencies
```bash
pip install diffusers transformers accelerate xformers torch torchvision
```

### 2. Verify Installation
```bash
python -c "import diffusers, transformers, accelerate; print('All packages installed successfully!')"
```

## 📁 Data Preparation

### 1. Choose Your Dataset Type

The training script supports two dataset types:

#### **Option A: RTMDataset** (Recommended for custom data)
- **Format**: Standard file-based dataset
- **Structure**: Images and labels in separate folders
- **Flexibility**: Easy to set up with your own data

#### **Option B: DocTamperDataset** (For LMDB format)
- **Format**: LMDB database format
- **Structure**: Pre-processed binary data
- **Performance**: Faster loading for large datasets

### 2. RTMDataset Structure
Organize your data as follows:
```
your_dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── labels/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── train.txt
```

### 3. DocTamperDataset Structure
Your data should be in LMDB format:
```
your_lmdb_files/
├── dataset1.lmdb
├── dataset2.lmdb
└── ...
```

### 4. Image Requirements
- **Format**: JPG/JPEG (RTMDataset) or LMDB (DocTamperDataset)
- **Size**: Any size (will be cropped to 512x512)
- **Channels**: RGB (3 channels)

### 5. Label Requirements
- **Format**: PNG (RTMDataset) or binary (DocTamperDataset)
- **Size**: Same as corresponding image
- **Values**: 
  - `0` (black) for authentic regions
  - `255` (white) for forgery regions
- **Channels**: Grayscale (will be converted to RGB)

### 6. Train Split File (RTMDataset only)
Create `train.txt` with image names (without extension):
```
image1
image2
image3
...
```

## ⚙️ Training Configuration

### 1. Choose Dataset Type
Edit `train_document_forgery.sh`:
```bash
# Choose your dataset type
DATASET_TYPE="rtm"        # For RTMDataset
# DATASET_TYPE="doctamper"  # For DocTamperDataset
```

### 2. RTMDataset Configuration
```bash
# Set your data paths for RTMDataset
DATA_ROOT="/path/to/your/dataset"
IMG_DIR="images"
LABEL_DIR="labels"
SPLIT_FILE="train.txt"
```

### 3. DocTamperDataset Configuration
```bash
# Set your LMDB paths for DocTamperDataset
LMDB_PATHS=("/path/to/your/lmdb1" "/path/to/your/lmdb2")
DATASET_MODE="train"  # Options: "train", "val", "test"
DATASET_SEED=42
```

### 4. Training Parameters
Key parameters you can adjust:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_TRAIN_STEPS` | 50000 | Total training steps |
| `TRAIN_BATCH_SIZE` | 2 | Batch size per GPU |
| `LEARNING_RATE` | 5e-05 | Learning rate |
| `GRADIENT_ACCUMULATION_STEPS` | 8 | Gradient accumulation |

### 5. Hardware Requirements
- **GPU**: At least 8GB VRAM (16GB+ recommended)
- **RAM**: 16GB+ system RAM
- **Storage**: 50GB+ free space for checkpoints

## 🚀 Start Training

### 1. Make Scripts Executable
```bash
chmod +x train_document_forgery.sh
chmod +x training/train_document_forgery.py
```

### 2. Configure Your Dataset

#### **For RTMDataset:**
```bash
# Edit train_document_forgery.sh
DATASET_TYPE="rtm"
DATA_ROOT="/path/to/your/dataset"
```

#### **For DocTamperDataset:**
```bash
# Edit train_document_forgery.sh
DATASET_TYPE="doctamper"
LMDB_PATHS=("/path/to/your/lmdb1" "/path/to/your/lmdb2")
```

### 3. Run Training
```bash
./train_document_forgery.sh
```

### 4. Monitor Training
```bash
# View TensorBoard logs
tensorboard --logdir logs/document_forgery

# Monitor GPU usage
nvidia-smi -l 1
```

## 📊 Training Monitoring

### 1. Loss Tracking
The training will log:
- **Train Loss**: Binary segmentation loss
- **Learning Rate**: Current learning rate
- **GPU Memory**: Memory usage
- **Dataset Type**: Which dataset is being used

### 2. Checkpoints
- Saved every `CHECKPOINTING_STEPS` (default: 10000)
- Location: `model-finetuned/document_forgery_sd/checkpoint-{step}`
- Can resume training from any checkpoint

### 3. Final Model
- Location: `model-finetuned/document_forgery_sd/`
- Contains: UNet, VAE, scheduler, text encoder

## 🔧 Troubleshooting

### 1. Out of Memory (OOM)
```bash
# Reduce batch size
TRAIN_BATCH_SIZE=1

# Increase gradient accumulation
GRADIENT_ACCUMULATION_STEPS=16

# Enable gradient checkpointing (already enabled)
```

### 2. Slow Training
```bash
# Enable mixed precision
--mixed_precision "fp16"

# Use xformers (already enabled)
--enable_xformers_memory_efficient_attention
```

### 3. Data Loading Issues

#### **RTMDataset Issues:**
```bash
# Check data paths
ls $DATA_ROOT/images/
ls $DATA_ROOT/labels/
cat $DATA_ROOT/train.txt

# Verify image formats
file $DATA_ROOT/images/*.jpg
```

#### **DocTamperDataset Issues:**
```bash
# Check LMDB files exist
ls /path/to/your/lmdb*.lmdb

# Verify LMDB files are readable
python -c "import lmdb; env = lmdb.open('/path/to/your/lmdb1', readonly=True); print('LMDB accessible')"
```

## 📈 Expected Results

### 1. Training Progress
- **Early**: Loss ~50-200
- **Mid**: Loss ~20-100  
- **Converged**: Loss ~5-50

### 2. Model Output
- **Authentic regions**: Predicted as black `[0, 0, 0]`
- **Forgery regions**: Predicted as white `[255, 255, 255]`

### 3. Training Time
- **50K steps**: ~8-12 hours on V100/A100
- **100K steps**: ~16-24 hours on V100/A100

## 🔄 Resume Training

### 1. From Latest Checkpoint
```bash
./train_document_forgery.sh --resume_from_checkpoint latest
```

### 2. From Specific Checkpoint
```bash
./train_document_forgery.sh --resume_from_checkpoint model-finetuned/document_forgery_sd/checkpoint-25000
```

## 🎯 Model Evaluation

### 1. Test Inference
```python
from gen2seg_sd_pipeline import gen2segSDPipeline
from PIL import Image

# Load your trained model
pipeline = gen2segSDPipeline.from_pretrained("model-finetuned/document_forgery_sd")

# Load test image
image = Image.open("test_image.jpg").convert("RGB")

# Generate segmentation
result = pipeline(image)
segmentation = result.prediction

# Save result
Image.fromarray(segmentation).save("segmentation.png")
```

### 2. Evaluation Metrics
- **Binary Accuracy**: Overall pixel accuracy
- **IoU**: Intersection over Union
- **Precision/Recall**: For forgery detection

## 📝 Customization

### 1. Different Loss Function
Edit `training/util/loss.py` to use different loss functions:
- `BinarySegmentationLoss`: For binary classification
- `InstanceSegmentationLoss`: For instance segmentation

### 2. Different Dataset
Modify `training/dataloaders/load.py` to support your dataset format.

### 3. Different Model
Change `PRETRAINED_MODEL` to use different SD models:
- `"stabilityai/stable-diffusion-2"`
- `"runwayml/stable-diffusion-v1-5"`
- `"CompVis/stable-diffusion-v1-4"`

## 🆘 Support

### 1. Common Issues
- **CUDA OOM**: Reduce batch size
- **Data loading errors**: Check file paths and formats
- **Training divergence**: Reduce learning rate

### 2. Debug Mode
```bash
# Add debug flags
--dataloader_num_workers 0
--mixed_precision "no"
```

### 3. Logs
Check logs in:
- `logs/document_forgery/` (TensorBoard)
- `model-finetuned/document_forgery_sd/` (checkpoints)

## 🎉 Success!

Once training completes, you'll have:
1. **Trained model** in `model-finetuned/document_forgery_sd/`
2. **Checkpoints** for resuming training
3. **Logs** for analysis
4. **Pipeline** ready for inference

Your model is now ready for document forgery detection! 🚀 