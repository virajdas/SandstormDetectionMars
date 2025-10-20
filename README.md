# Real-Time Hurricane Detection Using Satellite Imagery

This project aims to develop a machine learning-based system for detecting hurricanes in real time using satellite imagery. By training a model on historical hurricane images, the system learns to identify visual patterns associated with hurricane formation. It then applies this knowledge to live satellite data, captured either through screen recording or direct feed, to monitor and analyze atmospheric conditions.

The ultimate goal is to provide early visual indicators of hurricane development, supporting meteorological research and potentially aiding in disaster preparedness. This approach combines computer vision, real-time data processing, and deep learning to create a responsive and scalable detection tool.

The dataset used for training the model is the Hurricane Image Classification Dataset (https://images.cv/dataset/hurricane-image-classification-dataset), which contains labeled satellite images of hurricanes and non-hurricane weather patterns.

We are currently evaluating two approaches for the detection model:
- **TensorFlow-based CNN** for image classification.
- **YOLO (You Only Look Once)** for object detection and localization.

## Team Roles and Responsibilities

- **Viraj Das**: Project lead and model architect. Responsible for designing the detection pipeline and integrating real-time data capture.
- **Aman**: Data engineer. In charge of preprocessing the dataset, managing data pipelines, and ensuring data quality.
- **Adithi**: Research and evaluation. Focused on comparing model performance between TensorFlow and YOLO, and optimizing hyperparameters.
- **Ryan**: Deployment and visualization. Handles real-time inference, alert mechanisms, and building the user interface for monitoring.
