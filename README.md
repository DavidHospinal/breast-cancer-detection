# Breast Cancer Detection

Object detection system for identifying benign and malignant tumors in breast ultrasound images using the Roboflow Inference API.

![mamagif](https://github.com/user-attachments/assets/081cd2d4-970c-4349-a10a-cef9aa45cc43)


## Project Overview

This project implements a breast cancer detection pipeline using a pre-trained computer vision model from Roboflow Universe. The system analyzes ultrasound images and classifies detected tumors as either benign or malignant.

### Model Information

| Metric | Value |
|--------|-------|
| Model ID | breast-cancer-detection-phqga/5 |
| Model Type | Roboflow 2.0 Object Detection (Fast) |
| Checkpoint | COCOv6n |
| mAP@50 | 72.8% |
| Precision | 61.4% |
| Recall | 76.9% |
| Classes | benign, malignant |
| Training Date | 2022-08-29 |

## Development Environment

### Remote GPU Configuration

This project was developed using PyCharm Professional connected to Google Colab as a Jupyter Remote Server, leveraging the Tesla T4 GPU for accelerated processing.

**Technical Setup:**

- **IDE:** PyCharm Professional with Jupyter Remote Server integration
- **Runtime:** Google Colab managed runtime with GPU acceleration
- **GPU:** NVIDIA Tesla T4 (15GB VRAM)
- **File System:** Linux-based Colab environment (/bin, /content, /usr) accessible from PyCharm

This configuration allows seamless development in a professional IDE while utilizing cloud GPU resources for inference tasks.

## Features

- Real-time tumor detection in ultrasound images
- Bounding box visualization with confidence scores
- Interactive Gradio web interface
- Support for image upload and sample image selection
- Adjustable confidence threshold
- Detection statistics and class distribution analysis

## Installation

```bash
pip install inference-sdk gradio pillow requests matplotlib opencv-python numpy kagglehub
```

## Usage

### Quick Start with Inference SDK

```python
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="YOUR_API_KEY"
)

result = CLIENT.infer("your_image.jpg", model_id="breast-cancer-detection-phqga/5")
print(result)
```

### Running the Notebook

1. Open `breast_cancer_detection_v2.ipynb` in PyCharm or Google Colab
2. Execute cells sequentially from top to bottom
3. The Gradio interface will launch on port 7860
4. Upload an ultrasound image or select from sample images
5. Adjust confidence threshold as needed
6. Click "Ejecutar Inferencia" to run detection

## Project Structure

```
breast-cancer-detection/
    breast_cancer_detection_v2.ipynb   # Main notebook with Gradio interface
    requirements.txt                    # Python dependencies
    README.md                           # Project documentation
    sample_images/                      # Sample ultrasound images (optional)
```

## API Reference

### Inference Configuration

| Parameter | Type | Description |
|-----------|------|-------------|
| api_url | string | Roboflow serverless API endpoint |
| api_key | string | Your Roboflow API key |
| model_id | string | Model identifier (breast-cancer-detection-phqga/5) |
| confidence_threshold | float | Minimum confidence for detections (0.0-1.0) |

### Response Format

```json
{
  "predictions": [
    {
      "class": "malignant",
      "confidence": 0.727,
      "x": 438,
      "y": 138,
      "width": 158,
      "height": 103
    }
  ]
}
```

## Dataset
<img width="1578" height="703" alt="image" src="https://github.com/user-attachments/assets/7af1c2e4-0eb9-4dbc-a349-edbe3160a510" />

The model was trained on the Breast Cancer Detection Dataset from University of Ghana, available on Roboflow Universe.

- **Total Images:** 633
- **Classes:** 2 (benign, malignant)
- **Source:** Breast ultrasound imaging

For sample images used in this implementation, the BUSI (Breast Ultrasound Images) dataset was utilized for demonstration purposes.

## Citation

```bibtex
@misc{breast-cancer-detection-phqga_dataset,
    title = {Breast Cancer Detection Dataset},
    type = {Open Source Dataset},
    author = {University of Ghana},
    howpublished = {\url{https://universe.roboflow.com/university-of-ghana/breast-cancer-detection-phqga}},
    url = {https://universe.roboflow.com/university-of-ghana/breast-cancer-detection-phqga},
    journal = {Roboflow Universe},
    publisher = {Roboflow},
    year = {2022},
    month = {nov}
}
```

## License

This project uses the Breast Cancer Detection model from Roboflow Universe under its respective license terms.

## Author

Oscar David Hospinal R.
Pontificia Universidad Catolica de Chile  
oscardavid.hospinal@uc.cl



## Acknowledgments

- University of Ghana for the original dataset
- Roboflow for model hosting and inference API
- Google Colab for GPU runtime access
