# Fruits and Vegetable Classification

This project focuses on classifying various fruits and vegetables using deep learning techniques. It employs a Convolutional Neural Network (CNN) built with TensorFlow and Keras, trained on a dataset of labeled images.


## Introduction
The goal of this project is to build a machine learning model capable of accurately classifying images of fruits and vegetables. This model can be used in various applications, such as automated checkout systems in supermarkets or for educational purposes.

## Project Structure
- **App.py**: The main application script to run the trained model.
- **FV.h5**: The saved Keras model file.
- **requirements.txt**: A list of required Python packages.

## Setup and Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Mohammedazif/fruits-and-vegetable-classification.git
   cd fruits-and-vegetable-classification
   ```
2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application:**
   ```bash
   python App.py
   ```

## Model Architecture
The CNN model consists of multiple convolutional layers followed by max-pooling and fully connected layers. The architecture was optimized for accuracy and computational efficiency.

## Training
The model was trained on a dataset of fruits and vegetables images, with data augmentation techniques applied to increase the robustness of the model. Training was performed on a GPU to accelerate the process.

## Usage
After setting up the environment, the app can be run to classify images. The user inputs an image, and the model outputs the predicted class (e.g., apple, banana, carrot).


## Results
The model achieved a classification accuracy of **X%** on the test dataset. Detailed results and evaluation metrics are available in the training logs and Jupyter notebooks.

## Future Work
- **Expand Dataset**: Incorporate more fruit and vegetable classes to improve model generalization.
- **Model Optimization**: Explore different architectures and hyperparameters.
- **Deployment**: Deploy the model as a web or mobile application for real-time classification.
