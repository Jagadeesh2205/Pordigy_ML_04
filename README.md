## Hand Gesture Recognition

This repository contains a Python implementation for recognizing hand gestures using a Convolutional Neural Network (CNN). The code preprocesses images, builds a CNN model, trains it, and evaluates its performance.

## Overview

The project involves the following steps:
1. Loading and preprocessing hand gesture images.
2. Normalizing and encoding the images and labels.
3. Building and training a CNN model for gesture recognition.
4. Evaluating the model's performance and visualizing the training history.
5. Making predictions on new images.

## Features

- **Image Preprocessing:** Loads and preprocesses grayscale images.
- **CNN Model:** Implements a Convolutional Neural Network for gesture classification.
- **Data Augmentation:** Enhances the training data using various augmentation techniques.
- **Model Evaluation:** Provides accuracy and loss metrics along with visualizations.
- **Prediction Function:** Predicts gestures from new images.

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib
- scikit-learn
- OpenCV (optional, for image processing)

Install the required packages using `pip`:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/hand-gesture-recognition.git
    ```

2. Navigate into the project directory:

    ```bash
    cd hand-gesture-recognition
    ```

3. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Set the path to your dataset and sample image in the code.

2. Run the main script:

    ```bash
    python gesture_recognition.py
    ```

3. The script will:
   - Load and preprocess the dataset.
   - Train the CNN model.
   - Display training and validation accuracy/loss plots.
   - Make a prediction on a sample image.

## Code Structure

- `gesture_recognition.py`: Main script for loading data, training the model, and making predictions.
- `requirements.txt`: List of required Python packages.

## Example Usage

Update the `sample_img_path` variable in `gesture_recognition.py` with the path to an image you want to test. Run the script to see the predicted gesture.

```python
sample_img_path = 'path/to/your/image.png'
predicted_gesture = predict_image(model, sample_img_path)
print(f'Predicted gesture: {predicted_gesture}')
```

## Results

The script will output the test accuracy and display plots showing training and validation accuracy/loss. 

## Contributing

Feel free to contribute by submitting issues or pull requests. Please ensure that your code adheres to the project's coding standards and includes appropriate tests.

