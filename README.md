# Music Genre Classification using Neural Networks

This project is a music genre classification model that aims to predict the genre of a given music clip using deep learning techniques. The model is trained on the GTZAN dataset, which contains various music genres.

## Table of Contents
1. [Project Description](#project-description)
2. [Dataset](#dataset)
3. [Code](#code)
4. [Frontend](#frontend)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Contributing](#contributing)
8. [License](#license)
9. [Acknowledgments](#acknowledgments)
10. [Documentation](#documentation)

## Project Description

The goal of this project is to create a music genre classification model that can identify the genre of a given music clip accurately. The model is built using a combination of machine learning and deep learning techniques, specifically an Artificial Neural Network (ANN) and a Convolutional Neural Network (CNN).

## Dataset

The dataset used for training the model is the GTZAN dataset, which consists of audio clips from various music genres. This dataset is helpful in understanding sound and differentiating one song from another. The music clips are labeled with their respective genres, making it suitable for supervised learning tasks.

Dataset: [GTZAN Music Genre Classification](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification)

## Code

The provided code contains the implementation of the music genre classification model using Keras with TensorFlow backend. It consists of the following parts:

1. Loading and preprocessing the dataset
2. Building a simple ANN model
3. Training the ANN model
4. Managing overfitting with regularization
5. Implementing a Convolutional Neural Network (CNN) for genre classification
6. Evaluating the CNN model on the test set
7. Predicting the genre of new songs using the trained model

# Music Genre Classification Frontend

This part of the project contains the frontend code that allows users to interact with the trained music genre classification model. The frontend is built using Flask for the web server and HTML/CSS/JavaScript for the user interface.

### `app.py`

This file contains the Flask application that serves as the web server and handles user requests. It also includes functions for processing the audio file and making predictions using the trained model.

### Endpoints:

1. `/`: The index route that renders the main page where users can upload an audio file for prediction.
2. `/predict`: The route that receives the uploaded audio file, processes it, and returns the predicted music genre.

### `index.html`

This HTML file contains the user interface for the music genre classifier. It allows users to upload an audio file (in `.mp3` or `.wav` format) and see the predicted genre after processing.

### Elements:

1. **Upload Button**: Allows users to click and select an audio file to upload.
2. **Drop Area**: Users can drag and drop an audio file to upload it.
3. **Progress Bar**: Shows the progress of file upload and processing.
4. **Genre Result**: Displays the predicted genre after processing.
5. **Audio Player**: An HTML5 audio element to play the uploaded audio file.

### How to Use

1. Make sure you have the necessary dependencies installed to run the Flask application and serve the frontend.
2. Place the `Music_Genrel_CNN_Model.h5` file in the same directory as `app.py` to load the trained model.
3. Ensure that the `allowed_file` function in `app.py` supports the audio file formats you want to allow for prediction (e.g., `.mp3` and `.wav`).
4. Run the Flask application using the `app.py` script.
5. Access the application through a web browser and upload an audio file to see the predicted genre.


## Installation

To run the code, follow these steps:

1. Download the GTZAN dataset from the provided link and place it in the appropriate folder.
2. Install the required dependencies: `numpy`, `scikit-learn`, `tensorflow`, `keras`, `matplotlib`, `librosa`.
3. Run the code cells in the provided Python script or notebook.

## Usage

The code contains detailed explanations of each step, and you can use it to:

1. Load and preprocess the GTZAN dataset.
2. Build and train a simple ANN model for music genre classification.
3. Implement a regularized ANN model to manage overfitting.
4. Create a CNN model for genre classification.
5. Evaluate the trained CNN model on the test set.
6. Predict the genre of new songs using the trained model.

## Contributing

Contributions to this project are welcome! If you find any bugs or have improvements or new features to suggest, please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License. You can find the details in the LICENSE file.

## Acknowledgments

We would like to thank Kaggle and Andrada Olteanu for providing the GTZAN dataset, which makes this project possible.

## Documentation

For more detailed information about the project, you can refer to the documentation files provided in the repository or external documentation sources.

---

*Note: This README file provides an overview of the project and how to use the code. Make sure to execute the code in a Python environment with all the required dependencies installed.*
