# ArtEmis Image Captioning Project

## Overview
This project implements a Custom CNN + LSTM architecture designed to generate captions for the ArtEmis dataset. The system utilizes a custom Convolutional Neural Network (CNN) encoder built from scratch and an "Init-Inject" Long Short-Term Memory (LSTM) decoder. The implementation is optimized for training on consumer-grade hardware, specifically tested on an NVIDIA RTX 3050 GPU.

## Folder Structure
* **/ArtEmis**
    * `ArtEmis_Caption_Generation.ipynb`: Main Jupyter Notebook containing Exploratory Data Analysis (EDA).
    * `cnn_training.py`: Contains the main training loop and the model class definitions.
    * `image_to_tensor_conversion.py`: Preprocessing script to convert raw image files into PyTorch tensors.
    * `vocab_generation.py`: Preprocessing script to transform text data into JSON vocabulary files.
    * `create_embeddings_final.py`: Script for generating word embeddings via GloVe or TF-IDF.
    * `evaluate_model.py`: Script for calculating BLEU scores and performing qualitative evaluation.
    * `text_cnn/`: Directory for generated vocabulary and word-encoding files.
    * `cnn_lstm/`: Directory for saved model checkpoints and embedding weights.

## Setup Instructions

### Prerequisites
* Python 3.8 or higher
* PyTorch
* Torchvision
* NLTK
* NumPy
* Pandas
* TQDM

Ensure all dependencies are installed by running:
`pip install -r requirements.txt`

### Step 1: Data Preparation
1. Place the `artemis_dataset_release_v0.csv` file in the root directory.
2. Place the image dataset in the following local directory: `C:\Img10k`.

### Step 2: Image Preprocessing
To optimize training speed and reduce I/O bottlenecks, convert the JPEG images into PyTorch tensors:
`python image_to_tensor_conversion.py`
This will generate `.pt` files in `C:\Img10k_pt`.

### Step 3: Vocabulary Generation
Generate the required word mappings by running:
`python vocab_generation.py`
This produces `vocab.json` and `df_word_encoded.pkl` within the `text_cnn/` directory.

### Step 4: Embedding Generation
Prepare the embedding layer by running:
`python create_embeddings_final.py`
Note: Users must have the GloVe or FastText pre-trained vectors downloaded locally as per the script's configuration.

## Training
To execute the training process on a CUDA-enabled GPU:
`python cnn_training.py`

* The script monitors training and validation loss.
* The model state with the lowest validation loss is saved as `best_model.pth`.

## Evaluation
To assess model performance and generate sample captions:
`python evaluate_model.py`
Ensure that the `MODEL_PATH` variable within the script correctly references the saved `.pth` checkpoint.

## System Specifications and Optimization
* **VRAM Management:** The default batch size is set to 64, which is optimized for GPUs with 6GB of VRAM.
* **CPU Execution:** If a compatible GPU is unavailable, the batch size should be reduced to 8 in `cnn_training.py` to prevent system instability.
* **Normalization:** This project utilizes [0,1] scaling instead of standard ImageNet normalization. This supports the "from-scratch" training of the CNN, allowing it to adapt specifically to the artistic visual features present in the ArtEmis dataset.
* **Architecture Logic:** The "Init-Inject" method is used for the LSTM decoder, where the image features are injected as the initial hidden state. This ensures the model is grounded in the visual context before beginning the sequential text generation.
