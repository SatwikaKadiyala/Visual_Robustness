# Visual Model Robustness

A computer vision project that evaluates how different image conditions affect object detection performance. It uses a pretrained YOLO model to detect objects and compares the results under low light, rain, fog, and blur.

## Overview

Object detection models often perform well on clear images but may struggle when visibility is reduced. This project explores how changes in lighting, weather, and image quality affect object detection.

Users can upload an image, apply different visual conditions, and compare the model's predictions.

## Features

- Object Detection:Detects objects such as cars, pedestrians, bicycles, and buses using YOLO.
- Image Transformations:Simulates low light, rain, fog, and blur using image processing techniques.
- Performance Comparison:Compares object detection results between original and modified images.
- Image Visualization:Displays annotated images with bounding boxes and confidence scores.
- Model Evaluation:Includes utilities to calculate precision, recall, and F1-score using labeled images.
- Interactive Interface:Built with Streamlit for easy image upload and testing.

## Technologies Used

- Python:Core programming language
- PyTorch and YOLO:Deep learning and object detection
- OpenCV and Pillow:Image processing
- NumPy:Numerical operations
- Streamlit:Interactive web application
- Pytest:Automated testing

## Project Structure

```text
Visual_Robustness/
├── app.py
├── cli.py
├── benchmark.py
├── train.py
├── requirements.txt
├── visionlab/
│   ├── __init__.py
│   └── core.py
├── tests/
│   └── test_core.py
└── README.md
```

## Installation

1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/visual-model-robustness.git
cd visual-model-robustness
```

2. Create a virtual environment

```powershell
py -3.11 -m venv .venv
```

3. Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Running the Application

Start the Streamlit application:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### How to Use

1. Upload a street image containing vehicles, pedestrians, or other objects.
2. Run object detection on the original image.
3. Select a condition such as low light, rain, fog, or blur.
4. Run detection again on the modified image.
5. Compare the detected objects, bounding boxes, and confidence scores.

The pretrained YOLO model may download its weights automatically on the first run.

## Testing

Run the automated tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The project includes tests for image transformations and detection evaluation utilities.

## Project Goal

The goal is to understand how environmental conditions affect computer vision models and evaluate their reliability.

This project provides a foundation for further experiments involving model fine-tuning, data augmentation, and robustness evaluation on labeled datasets.

## Future Improvements

- Evaluate the model using a larger labeled image dataset.
- Compare different YOLO architectures.
- Fine-tune models using weather-based image augmentation.
- Measure detection performance using precision, recall, F1-score, and mAP across different conditions.

## Disclaimer

This is an experimental computer vision project intended for educational and research purposes. It is not designed for real-world safety-critical applications.
