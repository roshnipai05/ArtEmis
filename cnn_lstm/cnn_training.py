import os
import math
import random
import time
import json
import pickle
from pathlib import Path
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import numpy as np

# ================================================================
# 1) Encoder: Custom CNN (Simple & Shallow)
# ================================================================
class SimpleCNNEncoder(nn.Module):
    def __init__(self, image_feature_dim: int = 256, in_channels: int = 3):
        super().__init__()
        # 4 Blocks are enough for 128x128 input
        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)) # Forces output to (B, 256, 1, 1)
        )

        self.fc = nn.Linear(256, image_feature_dim)

    def forward(self, images: torch.Tensor):
        x = self.conv_blocks(images)
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.fc(x))
        return x


# ================================================================
# 2) Decoder: LSTM (Init-Inject Architecture)
# ================================================================
class LSTMDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        image_feature_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        pad_idx: int
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # CHANGE: LSTM input is now JUST the embedding size
        self.lstm = nn.LSTM(
            input_size=embedding_dim,  
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # CHANGE: Layers to map Image Features -> LSTM Initial State
        self.init_h = nn.Linear(image_feature_dim, hidden_dim)
        self.init_c = nn.Linear(image_feature_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, image_feats, captions):
        """
        image_feats: (Batch, Image_Dim)
        captions: (Batch, Seq_Len)
        """
        batch_size = image_feats.size(0)

        # 1. Initialize Hidden and Cell states from the Image
        # Unsqueeze and repeat for num_layers
        h0 = self.init_h(image_feats).unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)
        c0 = self.init_c(image_feats).unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)

        # 2. Embed text
        embeddings = self.embedding(captions) # (Batch, Seq_Len, Embed_Dim)
        
        # 3. Pass through LSTM using the image-derived initial states
        # No need to concatenate image to input anymore
        lstm_out, _ = self.lstm(embeddings, (h0, c0))
        
        # 4. Predict words
        logits = self.fc_out(self.dropout(lstm_out))
        return logits

    def generate_greedy(self, image_feats, max_len, start_token_id, end_token_id, device):
        batch_size = image_feats.size(0)

        # Initialize state from image ONCE
        h = self.init_h(image_feats).unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)
        c = self.init_c(image_feats).unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)

        generated = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        
        # Start with <start> token
        prev_word = torch.tensor([start_token_id] * batch_size, device=device)
        
        finished_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for t in range(max_len):
            embed = self.embedding(prev_word).unsqueeze(1) # (Batch, 1, Embed_Dim)
            
            output, (h, c) = self.lstm(embed, (h, c))
            logits = self.fc_out(output.squeeze(1))
            next_token = logits.argmax(dim=1)
            
            generated[:, t] = next_token
            prev_word = next_token

            # Update finished mask
            finished_mask |= (next_token == end_token_id)
            
            # Optimization: If all batch elements are finished, stop early
            if finished_mask.all():
                break
            
        return generated


# ================================================================
# 3) Masked Loss
# ================================================================
def masked_cross_entropy_loss(logits, targets, pad_idx):
    B, T, V = logits.size()
    logits_flat = logits.reshape(B * T, V)
    targets_flat = targets.reshape(B * T)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")
    loss_sum = loss_fn(logits_flat, targets_flat)
    non_pad = (targets_flat != pad_idx).sum().item()
    return loss_sum / non_pad if non_pad > 0 else torch.tensor(0.0, device=logits.device)


# ================================================================
# 4) Dataset — Loads PREPROCESSED .PT TENSORS
# ================================================================
class ArtEmisDataset(Dataset):
    def __init__(self, items, transform=None):
        self.items = items  # list of (original_jpeg_path, token_ids)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        jpeg_path, caption_ids = self.items[idx]

        # Convert JPEG path → .PT tensor path
        tensor_path = jpeg_path.replace("Img10k", "Img10k_pt")
        tensor_path = Path(tensor_path).with_suffix(".pt")
        
        # Load Tensor
        image = torch.load(tensor_path, weights_only=True)
        
        # Apply Augmentation (if any)
        if self.transform:
            image = self.transform(image)
            
        caption_ids = torch.tensor(caption_ids, dtype=torch.long)

        return image, caption_ids
    
# ================================================================
# 5) Collate
# ================================================================
def collate_fn(batch, pad_idx):
    images, captions = zip(*batch)
    images = torch.stack(images, dim=0)

    lengths = [c.size(0) for c in captions]
    max_len = max(lengths)
    padded = torch.full((len(captions), max_len), pad_idx, dtype=torch.long)
    for i, c in enumerate(captions):
        padded[i, :c.size(0)] = c

    inputs = padded
    targets = padded[:, 1:].clone() # Shifted right targets
    return images, inputs, targets, lengths

def collate_wrapper(batch):
    return collate_fn(batch, pad_idx=0)


# ================================================================
# 6) Training & Validation Loops
# ================================================================
def train_one_epoch(encoder, decoder, dataloader, optimizer, device, pad_idx):
    encoder.train()
    decoder.train()

    total_loss = 0
    n_batches = 0
    start_time = time.time()

    for batch_idx, (images, inputs, targets, lengths) in enumerate(dataloader):
        images = images.to(device)
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        
        # Forward
        image_feats = encoder(images)
        logits = decoder(image_feats, inputs) # inputs includes <start> ...
        
        # Ensure shapes match for loss
        if logits.size(1) != targets.size(1):
             min_len = min(logits.size(1), targets.size(1))
             logits = logits[:, :min_len, :]
             targets = targets[:, :min_len]

        loss = masked_cross_entropy_loss(logits, targets, pad_idx)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if (batch_idx + 1) % 50 == 0:
            print(f"   Batch {batch_idx+1}/{len(dataloader)} - Loss: {loss.item():.4f}")

    return total_loss / n_batches

def validate(encoder, decoder, dataloader, device, pad_idx):
    encoder.eval()
    decoder.eval()
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for images, inputs, targets, lengths in dataloader:
            images = images.to(device)
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            image_feats = encoder(images)
            logits = decoder(image_feats, inputs)
            
            if logits.size(1) != targets.size(1):
                 min_len = min(logits.size(1), targets.size(1))
                 logits = logits[:, :min_len, :]
                 targets = targets[:, :min_len]

            loss = masked_cross_entropy_loss(logits, targets, pad_idx)
            total_loss += loss.item()
            n_batches += 1
            
    return total_loss / n_batches if n_batches > 0 else 0


# ================================================================
# 7) Main
# ================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # HYPERPARAMETERS
    IMAGE_FEATURE_DIM = 256
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 512
    NUM_LAYERS = 1
    DROPOUT = 0.3
    BATCH_SIZE = 16  # Increased for efficiency on GPU
    LR = 3e-4
    NUM_EPOCHS = 30
    MAX_GEN_LEN = 30

    print("\n=== Loading Vocabulary ===")
    class SimpleVocab:
        def __init__(self, stoi, itos, pad_idx, start_token, end_token):
            self.stoi = stoi
            self.itos = itos
            self.pad_idx = pad_idx
            self.start_token = start_token
            self.end_token = end_token
            self.vocab_size = len(itos)

    with open("text_cnn/rev_vocab.json") as f:
        itos = json.load(f)
    with open("text_cnn/vocab.json") as f:
        stoi = json.load(f)

    vocab = SimpleVocab(stoi, itos, pad_idx=0, start_token=2, end_token=3)
    print("Vocab size:", vocab.vocab_size)

    print("\n=== Loading Dataset ===")
    with open("text_cnn/df_word_encoded.pkl", "rb") as f:
        all_items = pickle.load(f)
    print("Total items:", len(all_items))

    # SPLIT DATASET (90% Train / 10% Val)
    random.shuffle(all_items)
    split_idx = int(0.9 * len(all_items))
    train_items = all_items[:split_idx]
    val_items = all_items[split_idx:]
    
    print(f"Train size: {len(train_items)}")
    print(f"Val size:   {len(val_items)}")
    
    # === FIXED VALIDATION SAMPLES FOR VISUALIZATION ===
    # We pick 3 random items from val_items ONCE and freeze them
    fixed_val_samples = random.sample(val_items, 3)

    # AUGMENTATION (Train Only)
    train_transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        # Note: No ToTensor or Normalize because they are already tensors in [0, 1]
    ])

    train_dataset = ArtEmisDataset(train_items, transform=train_transform)
    val_dataset = ArtEmisDataset(val_items, transform=None)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_wrapper)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_wrapper)

    print("\n=== Building Models ===")
    encoder = SimpleCNNEncoder(IMAGE_FEATURE_DIM).to(device)
    decoder = LSTMDecoder(
        vocab_size=vocab.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        image_feature_dim=IMAGE_FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        pad_idx=vocab.pad_idx
    ).to(device)

    # LOAD EMBEDDINGS
    print("Loading pretrained embeddings...")
    # NOTE: Change this path to match whichever embedding you want to use (fasttext/glove/tfidf)
    emb_path = r"C:\Users\91887\Documents\ArtEmis\cnn_lstm\updated_embeddings\glove_matrix.npy"
    if os.path.exists(emb_path):
        emb_matrix = np.load(emb_path)
        print("Embedding matrix shape:", emb_matrix.shape)
        if emb_matrix.shape[0] != vocab.vocab_size:
             print(f"[WARN] Shape mismatch! Matrix {emb_matrix.shape[0]} vs Vocab {vocab.vocab_size}")
             print("Using random embeddings instead.")
        else:
            decoder.embedding.weight.data.copy_(torch.tensor(emb_matrix, dtype=torch.float32))
    else:
        print(f"[WARN] Embedding file not found at {emb_path}. Using random initialization.")

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LR)
    
    # SCHEDULER
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    print("\n=== Starting Training ===")
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        print(f"\n----- Epoch {epoch+1}/{NUM_EPOCHS} -----")
        
        # Train
        train_loss = train_one_epoch(encoder, decoder, train_loader, optimizer, device, vocab.pad_idx)
        
        # Validate
        val_loss = validate(encoder, decoder, val_loader, device, vocab.pad_idx)
        
        # Step Scheduler
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save Best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = "best_model.pth"
            torch.save({
                'epoch': epoch,
                'encoder_state': encoder.state_dict(),
                'decoder_state': decoder.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'vocab_size': vocab.vocab_size
            }, ckpt_path)
            print(f"Saved Best Model (Val Loss {val_loss:.4f})")

        # === FIXED SAMPLE PREDICTION ===
        print("\n[Fixed Sample Predictions]")
        encoder.eval()
        decoder.eval()
        
        for idx, (img_path, caption_ids) in enumerate(fixed_val_samples):
             # Load Tensor
            t_path = Path(img_path.replace("Img10k", "Img10k_pt")).with_suffix(".pt")
            img_tensor = torch.load(t_path).unsqueeze(0).to(device)
            
            with torch.no_grad():
                feat = encoder(img_tensor)
                gen_ids = decoder.generate_greedy(
                    feat, MAX_GEN_LEN, vocab.start_token, vocab.end_token, device
                )[0].cpu().tolist()
                
            # Decode
            if vocab.end_token in gen_ids:
                gen_ids = gen_ids[:gen_ids.index(vocab.end_token)]
                
            pred_cap = " ".join([vocab.itos[str(i)] for i in gen_ids if i not in [vocab.pad_idx, vocab.start_token]])
            # Only print true caption on first epoch to save screen space
            if epoch == 0:
                true_cap = " ".join([vocab.itos[str(i)] for i in caption_ids if i not in [vocab.pad_idx, vocab.start_token, vocab.end_token]])
                print(f"Sample {idx+1} Image: {Path(img_path).name}")
                print(f"   True: {true_cap}")

            print(f"   Pred (Ep {epoch+1}): {pred_cap}")

    print("\n=== Training Complete ===")