import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import time

# --- MOCK DATASET FOR SINGLE BATCH TESTING ---
# We create fake tensors to avoid path issues during this specific test
class MockDataset(Dataset):
    def __init__(self, size=16):
        self.size = size
        # Create 16 random "images" (tensors) of 128x128
        self.images = torch.randn(size, 3, 128, 128)
        # Create 16 identical captions: [Start, 5, 10, 15, End, Pad...]
        self.captions = torch.tensor([[2, 5, 10, 15, 3, 0, 0, 0]] * size, dtype=torch.long)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.images[idx], self.captions[idx]

# --- COPY OF YOUR ARCHITECTURE (Keep exactly as you provided) ---
class SimpleCNNEncoder(nn.Module):
    def __init__(self, image_feature_dim=256, in_channels=3):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(256, image_feature_dim)

    def forward(self, images):
        x = self.conv_blocks(images)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x

class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, image_feature_dim, hidden_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.init_h = nn.Linear(image_feature_dim, hidden_dim)
        self.init_c = nn.Linear(image_feature_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, image_feats, captions):
        h0 = self.init_h(image_feats).unsqueeze(0)
        c0 = self.init_c(image_feats).unsqueeze(0)
        embeddings = self.embedding(captions)
        lstm_out, _ = self.lstm(embeddings, (h0, c0))
        logits = self.fc_out(lstm_out)
        return logits

# --- THE TEST LOOP ---
def run_overfit_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Diagnostic on: {device}")

    # 1. Setup minimal data (ONE BATCH ONLY)
    dataset = MockDataset(size=16) # Batch size 16
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # 2. Setup Model
    # Small vocab for testing
    VOCAB_SIZE = 100 
    encoder = SimpleCNNEncoder(image_feature_dim=256).to(device)
    decoder = LSTMDecoder(vocab_size=VOCAB_SIZE, embedding_dim=128, image_feature_dim=256, hidden_dim=256).to(device)

    # 3. Aggressive Optimizer
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print("\n--- STARTING OVERFIT TEST (Target: Loss < 0.01) ---")
    encoder.train()
    decoder.train()

    for epoch in range(101):
        for images, captions in loader:
            images, captions = images.to(device), captions.to(device)
            
            inputs = captions # For testing, we just pass full captions (logic slightly simplified for mock)
            targets = captions # Predict itself for sanity check logic

            optimizer.zero_grad()
            
            # Forward
            feats = encoder(images)
            outputs = decoder(feats, inputs) # logits: (B, Seq, Vocab)
            
            # Reshape for loss
            # Shift targets: Input [A, B, C] -> Target [B, C, End]
            # Here we just check if it can memorize the sequence structure
            outputs = outputs[:, :-1, :] 
            targets = targets[:, 1:]

            loss = criterion(outputs.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
            
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
            if loss.item() < 0.01:
                print("\nSUCCESS: Model successfully overfitted on one batch.")
                print("Logic is sound. The issue is Model Capacity vs Dataset Complexity.")
                return

    print("\nFAILURE: Model could not overfit a single batch.")
    print("There is a bug in the Architecture (likely Gradient Flow or Tensor Shapes).")

if __name__ == "__main__":
    run_overfit_test()