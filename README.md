# Wildlife Object Detection using Faster R-CNN

Multi-species wildlife detection system for automated camera trap analysis using Faster R-CNN ResNet-101.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.5.3-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Overview

This project implements a deep learning-based object detection system for identifying and localizing three African wildlife species in camera trap imagery. The system uses Faster R-CNN with a ResNet-101 backbone, fine-tuned on annotated wildlife data to achieve high-precision detection suitable for conservation monitoring and population surveys.

## Problem Statement

Manual analysis of camera trap images is time-consuming and requires significant human effort. This system automates the detection and classification of wildlife species, enabling:
- Efficient processing of large-scale camera trap datasets
- Accurate species identification and counting
- Support for wildlife conservation and monitoring programs

## Dataset

- **Species**: 3 African wildlife classes
  - Oryx Gazella (Gemsbok)
  - Panthera Leo (Lion)
  - Phacochoerus Africanus (Warthog)
- **Size**: 500 images per species (1,500 total)
- **Annotations**: Bounding boxes in PASCAL VOC format
- **Split**: Train/Validation/Test

## Model Architecture

- **Base Model**: Faster R-CNN ResNet-101
- **Backbone**: ResNet-101 (pre-trained on COCO)
- **Framework**: TensorFlow 2.5.3 Object Detection API
- **Input Resolution**: 1024×1024 pixels
- **Training Strategy**: Fine-tuning with transfer learning

### Key Design Choices

- **High Resolution**: 1024×1024 input preserves fine-grained features (horns, manes, body shapes)
- **Deep Backbone**: ResNet-101 provides rich feature hierarchy for species discrimination
- **Transfer Learning**: COCO pre-training accelerates convergence on wildlife domain
- **Data Augmentation**: Horizontal flips, random crops, color adjustments for robustness

## Results

### Quantitative Performance

| Metric | Value |
|--------|-------|
| **mAP @ IoU=0.50** | **95.4%** |
| **mAP @ IoU=0.75** | 81.1% |
| **Recall** | 75% |
| **Classification Loss** | 0.088 |

### Qualitative Observations

- ✅ High-confidence detections on clear, unoccluded images
- ✅ Accurate species classification with no confusion between classes
- ✅ Well-aligned bounding boxes on medium-to-large subjects
- ⚠️ Reduced performance on heavily occluded or distant animals
- ⚠️ Occasional loose bounding boxes around extremities

### Per-Species Performance

The model demonstrates strong and consistent detection across all three species:
- **Oryx Gazella**: Excellent detection in open terrain, handles multiple individuals
- **Panthera Leo**: High confidence on both resting and moving lions
- **Phacochoerus Africanus**: Robust to background clutter and partial vegetation occlusion

## Repository Structure

```
wildlife-object-detection/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
│
├── data/
│   ├── label_map.pbtxt         # Class label definitions
│   ├── train/                  # Training images and annotations
│   ├── val/                    # Validation images and annotations
│   └── test/                   # Test images (unseen)
│
├── notebooks/
│   ├── 01_data_preparation.ipynb    # Data preprocessing and augmentation
│   ├── 02_model_training.ipynb      # Training pipeline
│   ├── 03_evaluation.ipynb          # Metrics and analysis
│   └── 04_inference.ipynb           # Inference on test images
│
├── models/
│   ├── pipeline.config          # Model configuration
│   ├── checkpoint/              # Trained model weights (step 20,000)
│   └── exported_model/          # Frozen SavedModel for deployment
│
├── src/
│   ├── preprocess.py           # Data preprocessing utilities
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   └── inference.py            # Inference utilities
│
└── results/
    ├── metrics/                # Training curves, mAP graphs
    ├── predictions/            # Annotated test images
    └── analysis/               # Qualitative analysis outputs
```

## Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.5.3
- CUDA 11.2 (for GPU support)

### Setup

```bash
# Clone the repository
git clone https://github.com/havilahchisom-lang/wildlife-object-detection.git
cd wildlife-object-detection

# Install dependencies
pip install -r requirements.txt

# Install TensorFlow Object Detection API
git clone https://github.com/tensorflow/models.git
cd models/research
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
pip install .
```

## Usage

### 1. Data Preparation

```bash
python src/preprocess.py --input_dir data/raw --output_dir data/processed
```

### 2. Training

```bash
python src/train.py \
  --pipeline_config_path models/pipeline.config \
  --model_dir models/checkpoint \
  --num_train_steps 20000
```

### 3. Evaluation

```bash
python src/evaluate.py \
  --model_dir models/checkpoint \
  --pipeline_config_path models/pipeline.config \
  --checkpoint_dir models/checkpoint
```

### 4. Inference

```python
import tensorflow as tf
from src.inference import load_model, run_detection

# Load the exported model
detect_fn = load_model('models/exported_model/saved_model')

# Run detection on an image
detections = run_detection(detect_fn, 'path/to/image.jpg')

# Visualize results
visualize_detections(image, detections, min_score_thresh=0.5)
```

## Training Configuration

- **Optimizer**: SGD with momentum (0.9)
- **Learning Rate**: Cosine decay from 0.04 to 0.0001
- **Batch Size**: 4
- **Training Steps**: 20,000
- **Checkpoint**: Step 20,000 (best validation performance)

## Model Export

The trained model is exported to TensorFlow SavedModel format for deployment:

```bash
python models/research/object_detection/exporter_main_v2.py \
  --input_type image_tensor \
  --pipeline_config_path models/pipeline.config \
  --trained_checkpoint_dir models/checkpoint \
  --output_directory models/exported_model
```

## Future Improvements

- [ ] Deploy model as REST API using FastAPI
- [ ] Implement real-time inference pipeline
- [ ] Add model quantization for edge deployment
- [ ] Expand dataset to include more species
- [ ] Optimize for small/distant object detection
- [ ] Add tracking capabilities for video sequences

## Technologies Used

- **TensorFlow 2.5.3**: Deep learning framework
- **TensorFlow Object Detection API**: Pre-built detection models and utilities
- **OpenCV**: Image processing
- **NumPy**: Numerical computations
- **Matplotlib**: Visualization

## Use Cases

- Wildlife population monitoring
- Conservation program assessment
- Species distribution analysis
- Automated camera trap processing
- Ecological research support

## Limitations

- Performance degrades on heavily occluded subjects
- Reduced accuracy on very small or distant animals
- Not optimized for real-time inference
- Limited to three species (extensible with retraining)

## Citation

If you use this work, please cite:

```
@misc{wildlife-detection-2026,
  author = {Chisom Havilah Ibeh},
  title = {Wildlife Object Detection using Faster R-CNN},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/havilahchisom-lang/wildlife-object-detection}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- TensorFlow Object Detection API for model architecture and utilities
- COCO dataset for pre-trained weights
- Liverpool John Moores University for academic support

## Contact

**Chisom Havilah Ibeh**  
Machine Learning Engineering Student  
[LinkedIn](https://linkedin.com/in/havilahibeh) | [GitHub](https://github.com/havilahchisom-lang)

---

*Built with TensorFlow | Part of MSc Artificial Intelligence coursework*

