import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import json
import os
import matplotlib.pyplot as plt
from pathlib import Path
from nltk.translate.bleu_score import sentence_bleu

# ================================================================
# 1. MODEL ARCHITECTURE (Must Match Training Exactly)
# ================================================================

class SimpleCNNEncoder(nn.Module):
    def __init__(self, image_feature_dim: int = 256, in_channels: int = 3):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(256, image_feature_dim)

    def forward(self, images: torch.Tensor):
        x = self.conv_blocks(images)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x

class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, image_feature_dim, hidden_dim, num_layers, dropout, pad_idx):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,  
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.init_h = nn.Linear(image_feature_dim, hidden_dim)
        self.init_c = nn.Linear(image_feature_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, image_feats, captions):
        # We only need forward for training, but kept for completeness
        h0 = self.init_h(image_feats).unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)
        c0 = self.init_c(image_feats).unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)
        embeddings = self.embedding(captions)
        lstm_out, _ = self.lstm(embeddings, (h0, c0))
        return self.fc_out(self.dropout(lstm_out))

    def generate_greedy(self, image_feats, max_len, start_token_id, end_token_id, device):
        batch_size = image_feats.size(0)
        h = self.init_h(image_feats).unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)
        c = self.init_c(image_feats).unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)
        
        generated = []
        prev_word = torch.tensor([start_token_id] * batch_size, device=device)
        
        for t in range(max_len):
            embed = self.embedding(prev_word).unsqueeze(1)
            output, (h, c) = self.lstm(embed, (h, c))
            logits = self.fc_out(output.squeeze(1))
            next_token = logits.argmax(dim=1)
            
            token_id = next_token.item()
            if token_id == end_token_id:
                break
                
            generated.append(token_id)
            prev_word = next_token
            
        return generated

# ================================================================
# 2. PREPROCESSING
# ================================================================

def process_image(image_path):
    """
    Resizes to 128x128 and converts to Tensor [0, 1].
    Ignores ImageNet normalization as per training strategy.
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor() # Converts [0, 255] -> [0.0, 1.0]
    ])
    
    return transform(img).unsqueeze(0) # Add batch dim: (1, 3, 128, 128)

# ================================================================
# 3. PREDICTION LOGIC
# ================================================================

def load_vocab(vocab_dir):
    try:
        with open(os.path.join(vocab_dir, "rev_vocab.json"), "r") as f:
            rev_vocab = json.load(f)
        with open(os.path.join(vocab_dir, "vocab.json"), "r") as f:
            vocab = json.load(f)
        return vocab, rev_vocab
    except FileNotFoundError:
        print(f"Error: Could not find vocab files in {vocab_dir}")
        exit(1)

def run_prediction(image_path, emotion, model_paths, vocab_dir, device, reference_caption=None):
    # 1. Load Vocab
    stoi, itos = load_vocab(vocab_dir)
    vocab_size = len(itos)
    
    # Special Tokens
    START_TOKEN = 2
    END_TOKEN = 3
    PAD_IDX = 0

    # 2. Process Image
    img_tensor = process_image(image_path)
    if img_tensor is None: return
    img_tensor = img_tensor.to(device)

    print(f"\n[{'='*20} PREDICTION REPORT {'='*20}]")
    print(f"Input Image: {image_path}")
    print(f"Input Emotion (Ignored): {emotion}")
    if reference_caption:
        print(f"Reference Caption: {reference_caption}")
    print("-" * 60)

    # 3. Iterate Over Models
    for model_path in model_paths:
        if not os.path.exists(model_path):
            print(f"Error: Model file not found: {model_path}")
            continue

        # Initialize Model
        # Hyperparams must match training!
        encoder = SimpleCNNEncoder(image_feature_dim=256).to(device)
        decoder = LSTMDecoder(
            vocab_size=vocab_size,
            embedding_dim=300,
            image_feature_dim=256,
            hidden_dim=512,
            num_layers=1,
            dropout=0.0,
            pad_idx=PAD_IDX
        ).to(device)

        # Load Weights
        checkpoint = torch.load(model_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state'])
        decoder.load_state_dict(checkpoint['decoder_state'])
        
        encoder.eval()
        decoder.eval()

        # Generate
        with torch.no_grad():
            feats = encoder(img_tensor)
            gen_ids = decoder.generate_greedy(
                feats, 
                max_len=30, 
                start_token_id=START_TOKEN, 
                end_token_id=END_TOKEN, 
                device=device
            )

        # Decode
        caption_words = [itos[str(idx)] for idx in gen_ids if str(idx) in itos]
        caption = " ".join(caption_words)
        
        print(f"Model: {Path(model_path).name}")
        print(f"Generated Caption: {caption}")
        
        # Calculate BLEU (Only if reference is provided)
        if reference_caption:
            ref_tokens = reference_caption.lower().split()
            # BLEU-1 Score
            score = sentence_bleu([ref_tokens], caption_words, weights=(1, 0, 0, 0))
            print(f"BLEU-1 Score: {score:.4f}")
            
        print("-" * 60)

    # 4. Show Image
    try:
        img = Image.open(image_path)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Generated: {caption}") # Shows caption from LAST model in loop
        plt.show()
    except Exception as e:
        print(f"Could not display image: {e}")

# ================================================================
# 4. MAIN
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate captions for ArtEmis images.")
    
    parser.add_argument("image_path", type=str, help="Path to the input .jpg image")
    parser.add_argument("emotion", type=str, help="Emotion label (e.g., 'joy') - Ignored by model")
    
    # Allow multiple models to be passed
    parser.add_argument("--models", nargs='+', default=["best_model.pth"], 
                        help="List of model checkpoints to use (default: best_model.pth)")
    
    parser.add_argument("--vocab_dir", type=str, default="text_cnn", 
                        help="Directory containing vocab.json and rev_vocab.json")

    # Optional: Reference caption for BLEU score
    parser.add_argument("--reference", type=str, default=None,
                        help="Ground truth caption (optional) to calculate BLEU score")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    run_prediction(args.image_path, args.emotion, args.models, args.vocab_dir, device, args.reference)