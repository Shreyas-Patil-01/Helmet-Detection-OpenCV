# Helmet Detection from Video using OpenCV and YOLO

This repository contains the code and resources for a helmet detection system that processes video streams to identify individuals wearing helmets. The system uses the YOLO (You Only Look Once) algorithm for object detection and a custom-trained PyTorch model to classify helmet usage.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Usage](#usage)
- [Installation](#installation)
- [Examples](#examples)
- [Results](#results)
- [Contributing](#contributing)


## Project Overview

### Description
This project offers a robust solution for detecting helmets in video feeds, enhancing safety protocols in various environments such as construction sites, traffic monitoring, and industrial areas. By leveraging the YOLO algorithm for object detection and a custom-trained PyTorch model, the system accurately distinguishes between individuals wearing helmets and those not wearing helmets.

### Objectives
- **Enhance Safety**: Automatically detect helmet usage to improve safety compliance.
- **Real-Time Monitoring**: Process video streams in real-time to identify safety violations instantly.
- **Easy Integration**:The system can be integrated into existing video surveillance setups.
- 
### Features
- **Real-Time Detection**: Capable of detecting helmets in video streams with minimal delay.
- **YOLO Integration**: Utilizes the powerful YOLO algorithm for efficient object detection.
- **Custom PyTorch Model**:Trained to accurately classify helmeted and non-helmeted individuals.
- **Configurable Settings**:Easily adjustable parameters for detection sensitivity, output format, and more.
- **User-Friendly**: Easy-to-use interface that simplifies the process of Helmet Detection.

### Benefits
- **User-Friendly Interface**: No need to learn complex algorithm.
- **Improved Efficiency**:Streamline safety checks, reducing the time and effort required for monitoring.
- **Improved Safety**:Automatically monitor helmet compliance without the need for manual checks.
- **Flexibility**: Adaptable to various use cases by fine-tuning the model on different datasets.
- **Scalable**: Suitable for various environments and easily scalable across multiple video feeds.

### Workflow
1. **Video Input**: The system takes a video stream as input.
2. **YOLO Detection**: YOLO detects objects (bikes, number plates) within the frames.
3. **Helmet Classification**: A custom PyTorch model classifies detected objects as helmeted or non-helmeted.
4. **Output**: The system outputs the video with visual annotations indicating helmet usage.

### Visual Representation
![Project Workflow](https://github.com/Shreyas-Patil-01/PromQL/blob/main/model_deployed_img.png)
![Project Workflow](https://github.com/Shreyas-Patil-01/PromQL/blob/main/Fine_tuned_model_details.png)
![Project Workflow](https://github.com/Shreyas-Patil-01/PromQL/blob/main/output_video.mp4)

These images illustrate the system's process and the output it generates.

### Examples
- **Construction Site Monitoring**: Automatically detect whether workers are wearing helmets.
- **Traffic Surveillance**: Identify helmet compliance among motorcyclists.
- **Industrial Safety**: Ensure helmet usage in high-risk industrial zones.


## Dataset
The dataset used for training contains labeled images indicating whether individuals are wearing helmets. It has been augmented and preprocessed to ensure a variety of scenarios are covered.

The dataset is sourced from Hugging Face and has been customized for this project.

## Model Training
The custom helmet classification model was trained using a combination of transfer learning and fine-tuning techniques. The steps involved in the training process include:
- **Data Preparation**: Augmenting and preprocessing images.
- **Model Selection**: Choosing a base model architecture suitable for object classification.
- **Training**: Fine-tuning the model on the helmet detection dataset.
- **Evaluation**: Assessing the model's accuracy and refining it based on performance metrics.

## Usage
To use this model, follow the steps below:
1. Install the necessary dependencies.
2. Load the pretrained YOLO and custom PyTorch models.
3. Run the detection script with a video file as input.
4. Receive the processed video with helmet detection annotations.
5. Access through its endpoint by requesting the model.

### Installation
Clone this repository and install the required dependencies:
git clone https://github.com/Shreyas-Patil-01/Helmet-Detection-OpenCV.git
cd Helmet-Detection-OpenCV
pip install -r requirements.txt

