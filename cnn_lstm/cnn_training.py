# caption_model.py
# PyTorch implementation: Custom CNN encoder + LSTM decoder (teacher forcing + masking)
# Requirements: torch, torchvision (for transforms), nltk (for BLEU if desired)
# For BLEU: pip install nltk ; then nltk.download('punkt')

import os
import math
import random
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import json
import pickle

# Optional: for evaluation
#from nltk.translate.bleu_score import corpus_bleu

# ---------------------
# 1) Encoder: Custom CNN
# ---------------------
class SimpleCNNEncoder(nn.Module):
    """
    Small custom CNN that reduces an image to a compact feature vector.
    Output dimension = image_feature_dim (e.g., 256)
    """
    def __init__(self, image_feature_dim: int = 256, in_channels: int = 3):
        super().__init__()
        # Convolutional blocks. Keep architecture small enough to train from scratch.
        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /4

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /8

            # Block 4
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # global average pool
        )

        # Final projection to required dim
        self.fc = nn.Linear(256, image_feature_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B, C, H, W)
        returns: (B, image_feature_dim)
        """
        x = self.conv_blocks(images)        # (B, 256, 1, 1)
        x = x.view(x.size(0), -1)           # (B, 256)
        feat = self.fc(x)                   # (B, image_feature_dim)
        feat = F.relu(feat)
        return feat

# ---------------------
# 2) Decoder: LSTM
# ---------------------
class LSTMDecoder(nn.Module):
    """
    Decoder that, at every time step, receives:
    - image_feature vector (same for all timesteps)
    - previous word embedding

    and outputs vocabulary logits for the next word.
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 200,
        image_feature_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.3,
        pad_idx: int = 0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        # Input to LSTM is [image_feat || word_embed]
        self.lstm_input_dim = image_feature_dim + embedding_dim

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(dropout)
        # Project LSTM hidden -> vocab logits
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, image_feats: torch.Tensor, captions: torch.Tensor, teacher_forcing_ratio: float = 1.0) -> torch.Tensor:
        """
        Vectorized forward pass for training.
        IMPORTANT:
            This version assumes teacher_forcing_ratio == 1.0 (standard teacher-forcing).
            It generates predictions for tokens 1..T-1 in a single batched LSTM call.

        Args:
            image_feats: (B, image_feature_dim)
            captions: (B, T) including <start> token at index 0
        Returns:
            logits: (B, T-1, vocab_size)
        """
        batch_size, seq_len = captions.size()
        device = captions.device

        # (B, T, embed_dim)
        embeddings = self.embedding(captions)

        # We shift embeddings to get previous-word embeddings for each step 1..T-1:
        # inputs_emb = embeddings[:, :-1, :] → predicted outputs for tokens 1..T-1
        prev_word_embeds = embeddings[:, :-1, :]    # (B, T-1, E)

        # Repeat image features across sequence length
        # image_feats: (B, F) -> (B, 1, F) --> repeated -> (B, T-1, F)
        image_feats_rep = image_feats.unsqueeze(1).repeat(1, seq_len - 1, 1)

        # Concatenate image features + previous-word embedding
        # LSTM input: (B, T-1, F+E)
        lstm_inputs = torch.cat([image_feats_rep, prev_word_embeds], dim=2)

        # Initialize hidden states (zeros)
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=device)
        c0 = torch.zeros_like(h0)

        # Run LSTM over the entire sequence at once
        lstm_out, _ = self.lstm(lstm_inputs, (h0, c0))    # (B, T-1, hidden_dim)

        # Dropout + linear projection to vocab logits
        lstm_out = self.dropout(lstm_out)
        logits = self.fc_out(lstm_out)                    # (B, T-1, vocab_size)

        return logits

    def generate_greedy(self, image_feats: torch.Tensor, max_len: int, start_token_id: int, end_token_id: int, device: torch.device):
        """
        Greedy decode for inference.
        Returns token ids (B, <=max_len)
        """
        batch_size = image_feats.size(0)
        h = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=device)
        c = torch.zeros_like(h)

        generated = torch.full((batch_size, max_len), fill_value=end_token_id, dtype=torch.long, device=device)
        prev_word = torch.full((batch_size,), fill_value=start_token_id, dtype=torch.long, device=device)
        prev_embed = self.embedding(prev_word)

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for t in range(max_len):
            lstm_in = torch.cat([image_feats, prev_embed], dim=1).unsqueeze(1)  # (B,1,input_dim)
            out, (h, c) = self.lstm(lstm_in, (h, c))
            out = out.squeeze(1)
            logits = self.fc_out(out)
            next_token = logits.argmax(dim=1)  # (B,)
            generated[:, t] = next_token
            prev_embed = self.embedding(next_token)
            finished = finished | (next_token == end_token_id)
            if finished.all():
                break

        return generated  # token ids

# ---------------------
# 3) Utility: Masked loss function wrapper
# ---------------------
def masked_cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    logits: (B, T, V)
    targets: (B, T) - targets correspond to the "next" tokens for each input position.
             e.g., given input tokens [<start>, w1, w2], targets are [w1, w2, <end>] (padded as necessary)
    We flatten and use ignore_index=pad_idx
    """
    B, T, V = logits.size()
    logits_flat = logits.reshape(B * T, V)
    targets_flat = targets.reshape(B * T)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction='sum')  # we'll normalize manually
    loss_sum = loss_fn(logits_flat, targets_flat)
    # normalize by number of non-pad tokens
    non_pad = (targets_flat != pad_idx).sum().item()
    if non_pad == 0:
        return torch.tensor(0.0, device=logits.device)
    return loss_sum / non_pad

# ---------------------
# 4) Example Dataset & Collate (skeleton)
# ---------------------
class ArtEmisDataset(Dataset):
    """
    You must implement this to return:
        image_tensor: (3,H,W) float in [0,1] or normalized
        caption_ids: LongTensor (T) with <start> ... <end>
    The collate below will pad variable-length captions.
    """
    def __init__(self, items, transform=None):
        """
        items: list of dicts or tuples with (image_path, caption_ids)
        """
        self.items = items
        self.transform = transform or transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        image_path, caption_ids = self.items[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        caption_ids = torch.tensor(caption_ids, dtype=torch.long)
        return image, caption_ids

def collate_fn(batch, pad_idx: int):
    """
    batch: list of (image, caption_ids)
    returns:
        images: (B,3,H,W)
        captions_padded: (B, T_max)
        targets: (B, T_max-1)  -> next tokens
    """
    images, captions = zip(*batch)
    images = torch.stack(images, dim=0)
    lengths = [c.size(0) for c in captions]
    max_len = max(lengths)
    padded = torch.full((len(captions), max_len), fill_value=pad_idx, dtype=torch.long)
    for i, c in enumerate(captions):
        padded[i, :c.size(0)] = c

    # Inputs to decoder are padded (B, T)
    # Targets are the next tokens: padded[:,1:] (predict 1..T-1). We'll pad to same length (T-1).
    inputs = padded
    targets = padded[:, 1:].clone()
    return images, inputs, targets, lengths

# ---------------------
# 5) Training + Validation loops
# ---------------------
def train_one_epoch(
    encoder: nn.Module,
    decoder: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pad_idx: int,
    teacher_forcing_ratio: float = 1.0
) -> float:
    encoder.train()
    decoder.train()
    total_loss = 0.0
    n_batches = 0

    for images, inputs, targets, lengths in dataloader:
        images = images.to(device)
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        image_feats = encoder(images)  # (B, image_feature_dim)
        logits = decoder(image_feats, inputs, teacher_forcing_ratio=teacher_forcing_ratio)  # (B, T-1, V)
        loss = masked_cross_entropy_loss(logits, targets, pad_idx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / (n_batches if n_batches else 1)


def validate(
    encoder: nn.Module,
    decoder: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    pad_idx: int,
    vocab,
    max_gen_len: int = 30
) -> Tuple[float, List[List[int]], List[List[int]]]:
    encoder.eval()
    decoder.eval()
    total_loss = 0.0
    n_batches = 0

    references = []
    hypotheses = []

    with torch.no_grad():
        for images, inputs, targets, lengths in dataloader:
            images = images.to(device)
            inputs = inputs.to(device)
            targets = targets.to(device)

            image_feats = encoder(images)
            logits = decoder(image_feats, inputs, teacher_forcing_ratio=0.0)  # feed 0 -> pure teacher forcing off
            loss = masked_cross_entropy_loss(logits, targets, pad_idx)
            total_loss += loss.item()
            n_batches += 1

            # Generate greedy captions
            batch_size = images.size(0)
            gen_ids = decoder.generate_greedy(
                image_feats, max_len=max_gen_len,
                start_token_id=vocab.start_token, end_token_id=vocab.end_token, device=device
            )  # (B, max_len)

            # Collect references/hypotheses for BLEU (token lists)
            for i in range(batch_size):
                # reference: remove pad and start token, convert to ints (use inputs or original targets)
                # Assume the dataset original caption had <start>.. <end>
                # Build reference tokens (list of lists for nltk)
                target_seq = inputs[i].cpu().tolist()  # includes <start>
                # trim padding and <start>
                # find first pad or end token
                try:
                    end_pos = target_seq.index(vocab.end_token)
                except ValueError:
                    end_pos = len(target_seq)
                ref = [vocab.itos[w] for w in target_seq[1:end_pos]]  # skip <start>
                if len(ref) == 0:
                    ref = ['']  # fallback
                references.append([ref])  # list of reference lists

                # hypothesis: convert generated until end token
                gen_list = gen_ids[i].cpu().tolist()
                try:
                    epos = gen_list.index(vocab.end_token)
                    gen_trim = gen_list[:epos]
                except ValueError:
                    gen_trim = gen_list
                hyp = [vocab.itos[w] for w in gen_trim]
                hypotheses.append(hyp)

    avg_loss = total_loss / (n_batches if n_batches else 1)
    # compute BLEU-4 (simple)
    bleu_score = corpus_bleu(references, hypotheses) if hypotheses else 0.0

    return avg_loss, bleu_score, references, hypotheses

# ---------------------
# 6) Example usage
# ---------------------
if __name__ == "__main__":
    # ----------------------------
    # HYPERPARAMS - edit as needed
    # ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMAGE_FEATURE_DIM = 256
    EMBEDDING_DIM = 200
    HIDDEN_DIM = 512
    NUM_LAYERS = 1
    DROPOUT = 0.3
    BATCH_SIZE = 32
    LR = 1e-3
    NUM_EPOCHS = 50
    TEACHER_FORCING_RATIO = 0.8
    MAX_GEN_LEN = 30
    
    # ----------------------------
    # Vocabulary placeholder (you must provide this structure)
    # ----------------------------
    class SimpleVocab:
        def __init__(self, stoi, itos, pad_idx, start_token, end_token):
            self.stoi = stoi
            self.itos = itos
            self.pad_idx = pad_idx
            self.start_token = start_token
            self.end_token = end_token
            self.vocab_size = len(itos)
    
    with open("text_cnn/rev_vocab.json", "r") as f:
        itos = json.load(f)
    with open("text_cnn/vocab.json", "r") as f:
        stoi = json.load(f)

    vocab = SimpleVocab(stoi, itos, pad_idx=0, start_token=1, end_token=2)

    # ----------------------------
    # Example dataset stub (replace with real dataset)
    # Each item: (image_path, caption_ids_list)
    # ----------------------------

    with open("text_cnn/df_word_encoded.pkl", "rb") as f:
        example_items = pickle.load(f)

    # create datasets/dataloaders
    train_dataset = ArtEmisDataset(example_items, transform=transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
    ]))
    # if no data, this will be empty - replace example_items with real dataset.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_idx=vocab.pad_idx))

    # ----------------------------
    # Model, optimizer
    # ----------------------------
    encoder = SimpleCNNEncoder(image_feature_dim=IMAGE_FEATURE_DIM).to(device)
    decoder = LSTMDecoder(
    vocab_size=vocab.vocab_size,
    embedding_dim=EMBEDDING_DIM,
    image_feature_dim=IMAGE_FEATURE_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    pad_idx=vocab.pad_idx).to(device)

    # Load your pretrained embeddings
    embedding_matrix_path = r"C:\Users\91887\Documents\ArtEmis\cnn_lstm\updated_embeddings\fasttext_matrix.npy"
    emb_matrix = np.load(embedding_matrix_path)
    emb_tensor = torch.tensor(emb_matrix, dtype=torch.float32)

    decoder.embedding.weight.data.copy_(emb_tensor)

    print("Loaded pretrained embeddings:", embedding_matrix_path)
    

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=LR)

    # Example training loop (very small illustrative)
    for epoch in range(NUM_EPOCHS):
        if len(train_loader) == 0:
            print("No training data found in example_items. Populate dataset and restart.")
            break
        train_loss = train_one_epoch(
            encoder, decoder, train_loader, optimizer, device,
            pad_idx=vocab.pad_idx, teacher_forcing_ratio=TEACHER_FORCING_RATIO
        )
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train loss: {train_loss:.4f}")

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'encoder_state': encoder.state_dict(),
            'decoder_state': decoder.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }, f"checkpoint_epoch_{epoch+1}.pth")

    # After training: inference on single image
    # img = load and preprocess -> tensor (1,3,H,W)
    # img_feat = encoder(img.to(device))
    # gen_ids = decoder.generate_greedy(img_feat, max_len=MAX_GEN_LEN, start_token_id=vocab.start_token, end_token_id=vocab.end_token, device=device)
    # print("Generated tokens:", [vocab.itos[i] for i in gen_ids[0].cpu().tolist()])
