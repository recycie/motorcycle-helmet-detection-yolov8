# Helmet Detection System

## Overview

Helmet detection system developed for use within Buriram Rajabhat University. It utilizes YOLOv8, a state-of-the-art object detection algorithm, to identify and track helmet usage in real-time video streams. This system comprises two main components:

1. A high-performance API server for real-time detection
2. A user-friendly Flask web application for system interaction and result visualization

Built with Python 3.10.13, PyTorch, and optimized for NVIDIA CUDA GPU acceleration, this system offers robust performance for critical safety monitoring applications.

## System Requirements

- Python 3.10.13
- Anaconda for Python environment management
- NVIDIA CUDA-compatible GPU (for optimal performance)

## Installation Guide

### 1. Set Up the Environment

First, ensure you have [Anaconda](https://www.anaconda.com/products/individual) installed on your system. Then, set up the project environment:

```bash
# Create a new conda environment
conda create -n helmet-detection python=3.10.13

# Activate the environment
conda activate helmet-detection

# Clone the repository (assuming you have git installed)
git clone https://github.com/your-username/helmet-detection-system.git
cd helmet-detection-system

# Install required packages
pip install -r requirements.txt
```

### 2. Download Model Files

Download the YOLOv8 model files from [Google Drive](https://drive.google.com/file/d/1xbIqdnXYr2Q1xcX0VGcrHkaJ2bhrj209/view?usp=sharing) and place them in the `models/` directory.

## Usage Instructions

### 1. Launch the Detection Server

Start the real-time detection API server:

```bash
python detection.py
```

The server will display the API endpoint (typically `http://localhost:8001`) upon successful startup.

### 2. Run the Web Application

In a new terminal window, start the Flask web application:

```bash
python app/app.py
```

### 3. Access the System

- **Web Interface**: Open your browser and navigate to `http://localhost:5000/`
- **API**: Send requests to `http://localhost:8001` (or the endpoint provided by the detection server)

## Testing

For system testing, you can use this [sample video](https://drive.google.com/file/d/1uZ9V5SmZoQZLyYMHZGk2Z4srZShCjE8Q/view?usp=sharing) from Google Drive.


## Contact

For any queries or support, please contact:

Pete - kasidetpete@gmail.com

Project Link: [https://github.com/your-username/helmet-detection-system](https://github.com/your-username/helmet-detection-system)
