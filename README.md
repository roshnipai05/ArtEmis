==========================================================================
ARTEMIS IMAGE CAPTIONING - PROJECT README

OVERVIEW

This project implements a Custom CNN + LSTM architecture to generate captions
for the ArtEmis dataset. It features a "from-scratch" CNN encoder and an
"Init-Inject" LSTM decoder, optimized for training on consumer GPUs (RTX 3050).

FOLDER STRUCTURE

/ArtEmis
|-- ArtEmis_Caption_Generation.ipynb   # Main Jupyter Notebook (EDA) 
|-- cnn_training.py                    # Main Training Loop & Model Class
|-- image_to_tensor_conversion.py      # Preprocessing Script (Images -> Tensors)
|-- vocab_generation.py                # Preprocessing Script (Text -> JSON)
|-- create_embeddings_final.py         # Embedding Generation (GloVe/TF-IDF)
|-- evaluate_model.py                  # BLEU/Qualitative Eval Script
|-- text_cnn/                          # Generated Vocabulary files
|-- cnn_lstm/                          # Saved Checkpoints & Embeddings
|-- README.txt                         # This file

SETUP INSTRUCTIONS

Prerequisites: Python 3.8+, PyTorch, Torchvision, NLTK, NumPy, Pandas, TQDM 
Set up a virtual environment and install all libraries/frameworks from the requirmenets.txt 

Step 1: Data Preparation

Place the 'artemis_dataset_release_v0.csv' in the root folder.

Place your images in 'C:\Img10k'.

Step 2: Generate Tensors (Crucial for Speed)

Run: python image_to_tensor_conversion.py

This converts JPEGs to .pt files in 'C:\Img10k_pt'.

Step 3: Generate Vocabulary

Run: python vocab_generation.py

This creates 'vocab.json' and 'df_word_encoded.pkl'.

Step 4: Generate Embeddings

Run: python create_embeddings_final.py

Ensure you have downloaded GloVe/FastText vectors.

TRAINING

To start training on the GPU:

Run: python cnn_training.py

Monitor the console for Loss values.

Checkpoints are saved as 'best_model.pth'.

EVALUATION

To generate captions and calculate BLEU scores:

Run: python evaluate_model.py

Ensure 'MODEL_PATH' in the script points to your .pth file.

SYSTEM NOTES

The code uses 'Batch Size = 64' optimized for 6GB VRAM.

If running on CPU, reduce Batch Size to 8 in 'cnn_training.py'.

Normalization is set to [0,1] scaling (not ImageNet) to support scratch training.