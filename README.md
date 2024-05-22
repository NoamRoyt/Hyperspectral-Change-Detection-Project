# Hyperspectral-Change-Detection-Project
This project focuses on detecting anomalies in hyperspectral images using two methods: Chronochrome and Covariance Equalization. These techniques analyze spectral images captured at different times to identify significant changes. Our goal is to evaluate and enhance the effectiveness of these methods in detecting anomalies.

# Project Overview
Hyperspectral imaging provides detailed spectral information for each pixel, with images consisting of multiple wavelength bands. In this project, we use hyperspectral images where each pixel contains 126 different wavelength bands, captured a few hours apart.

# Methods

## Chronochrome
Chronochrome leverages spectral and temporal correlations to identify anomalies, making it robust against environmental variations.

Steps:

-Normalization: Adjust data to have a mean of zero.

-Covariance Matrix Calculation: Understand variability within and between images.

-Temporal Prediction: Predict spectral characteristics of one image based on the other.

-Difference Analysis: Compare predicted and actual data to detect anomalies.

## Covariance Equalization
Covariance Equalization normalizes the covariance of image data to improve change detection accuracy, especially in environments with significant background variations.

Steps:

-Normalization: Adjust data to have a mean of zero.

-Covariance Matrix Calculation: Calculate covariance matrices for the images.

-Difference Calculation: Define difference cubes and covariance matrices to highlight changes.

-Error Analysis: Measure the difference between predicted and actual data to detect anomalies.


# Implementation
The project uses Python and NumPy for numerical computations:

-Data Loading: Load hyperspectral images from the RIT dataset.

-Normalization: Normalize the data.

-Covariance Matrix Calculation: Calculate covariance matrices.

-Target Injection: Simulate changes by injecting a target into the second image.

-Anomaly Detection: Detect anomalies using the calculated difference and its covariance matrix.

-ROC Analysis: Build ROC curves to evaluate detection performance.


# Expanded Research
We applied a new normalization method that normalizes each pixel by subtracting the average value of its neighbors, enhancing local contrast and improving anomaly detection robustness.

# Conclusion
Covariance Equalization performs slightly better than Chronochrome in detecting anomalies. The new normalization method significantly improves the performance of both algorithms, making them more robust and accurate in identifying changes in hyperspectral imaging data.
