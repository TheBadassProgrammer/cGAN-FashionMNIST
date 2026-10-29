# Controlled Catalog Synthesizer (cGAN)

**Track A: Generative AI (Unit 3)**  
**Project:** Conditional Image Generation on Fashion-MNIST  

## Project Overview
The **Controlled Catalog Synthesizer** is a lightweight Conditional Generative Adversarial Network (cGAN) built using PyTorch. Unlike standard GANs that generate random images from noise, this conditional architecture acts like an API endpoint: it takes a specific class label (e.g., "Ankle Boot" or "T-Shirt") as input and strictly generates that requested item. 

This project demonstrates how to eliminate "AI hallucinations" and exert strict programmatic control over generative outputs without relying on massive, computationally expensive generalist models.

## Problem Statement & Motivation
*   **The Problem with Standard GANs:** They rely on random noise, making it impossible to guarantee a specific output class.
*   **The Problem with Large Models (e.g., DALL-E/ChatGPT):** They require heavy prompt engineering, suffer from unpredictable hallucinations (adding unwanted backgrounds or lighting), and cannot be easily run on edge devices or strictly private on-premise servers.
*   **Our Solution:** A highly constrained, specialized cGAN that maps discrete class labels to specific visual structures in milliseconds, trainable on a standard Colab GPU.

## System Architecture
The system consists of two competing neural networks:
1.  **The Generator (The Forger):** Takes a random noise vector (`z`) and a mathematical embedding of the requested class label. It processes these through `Linear` and `LeakyReLU` layers, outputting a 28x28 grayscale image via a `Tanh` activation function.
2.  **The Discriminator (The Detective):** Takes a 28x28 image and a class label embedding. It processes the flattened image through `Linear` and `LeakyReLU` layers (using `Dropout` to prevent overfitting/memorization) and outputs a probability score via a `Sigmoid` function.

**Loss & Optimization:** 
*   Binary Cross-Entropy Loss (BCE)
*   Adam Optimizer (Learning Rate: 0.0002)

## Dataset
**Fashion-MNIST**
*   **Image Size:** 28x28 pixels (Grayscale)
*   **Categories:** 10 distinct clothing classes (e.g., T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).
*   *Note: The dataset is automatically downloaded via `torchvision.datasets` upon running the script.*

## Installation & Setup

### Prerequisites
Ensure you have Python 3.8+ installed. The project requires the following libraries:
```bash
pip install torch torchvision matplotlib numpy jupyter
```

### Running the Project
1. Clone this repository to your local machine (or upload the notebook to Google Colab).
2. Open the Jupyter Notebook:
```bash
jupyter notebook cGAN_FashionMNIST.ipynb
```
3. Run all cells in the notebook. 
    * The script will automatically download the dataset, initialize the models, and run the adversarial training loop for 50 epochs.
    * Training takes approximately 10-15 minutes on a standard GPU (e.g., Google Colab T4).
4. View the final evaluation cell to see the strictly controlled generated outputs for your chosen target class.

## Experimental Results
During the evaluation phase, the Generator weights are locked (`gen.eval()`), and a specific class tensor (e.g., `Label 9` for Ankle Boots) is passed to the network alongside random noise. The model successfully synthesizes the requested item shape without relying on pre-existing backgrounds, validating the conditional generative pipeline.

## Team Members
*   **Nitin Tripathi** (Roll No: 2400290120168)
*   **Rishabh Raj** (Roll No: 2400290120203)
