#!/usr/bin/env python3
"""
Example: Using DeepBioP with PyTorch for DNA Sequence Classification

This example demonstrates how to use DeepBioP encodings with PyTorch models
for binary classification of DNA sequences.

Requirements:
    pip install deepbiop torch numpy

Usage:
    python examples/pytorch_integration.py
"""

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("PyTorch not installed. Install with: pip install torch")
    exit(1)

# Will work once Python bindings are implemented (T028-T032)
try:
    import deepbiop as dbp

    BINDINGS_AVAILABLE = True
except ImportError:
    print("DeepBioP Python bindings not yet available.")
    print("This example will work after completing tasks T028-T032.")
    BINDINGS_AVAILABLE = False


# ==============================================================================
# Example 1: CNN for DNA Sequence Classification
# ==============================================================================


class DNADataset(Dataset):
    """PyTorch Dataset for DNA sequences."""

    def __init__(self, sequences, labels, encoder):
        """
        Args:
            sequences: List of DNA sequences (bytes)
            labels: List of labels (0 or 1)
            encoder: DeepBioP encoder instance
        """
        self.sequences = sequences
        self.labels = labels
        self.encoder = encoder

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Encode sequence
        encoded = self.encoder.encode(self.sequences[idx])

        # Convert to tensor
        X = torch.from_numpy(encoded)
        y = torch.tensor(self.labels[idx], dtype=torch.long)

        return X, y


class DNAClassifierCNN(nn.Module):
    """1D CNN for DNA sequence classification."""

    def __init__(self, num_classes=2):
        super().__init__()

        self.conv_layers = nn.Sequential(
            # Input: [batch, 4, length] (one-hot encoded)
            nn.Conv1d(4, 64, kernel_size=8, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=8, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=8, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x shape: [batch, length, 4]
        # Permute to [batch, 4, length] for Conv1d
        x = x.permute(0, 2, 1)

        # Convolutional layers
        features = self.conv_layers(x)

        # Classification
        logits = self.classifier(features)

        return logits


def train_cnn_classifier():
    """Train a CNN classifier on DNA sequences."""

    if not BINDINGS_AVAILABLE:
        print("Skipping CNN training - bindings not available")
        return

    print("=" * 70)
    print("Example 1: CNN for DNA Sequence Classification")
    print("=" * 70)

    # Synthetic data (replace with real data)
    sequences = [
        b"ACGTACGTACGTACGT" * 10,  # 160 bp, class 0
        b"TTGGCCAATTGGCCAA" * 10,  # 160 bp, class 0
        b"AAAACCCCGGGGTTTT" * 10,  # 160 bp, class 1
        b"CGCGCGCGCGCGCGCG" * 10,  # 160 bp, class 1
    ] * 25  # 100 samples

    labels = [0, 0, 1, 1] * 25

    # Create encoder
    encoder = dbp.OneHotEncoder("dna", "skip")

    # Create dataset
    dataset = DNADataset(sequences, labels, encoder)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Create model
    model = DNAClassifierCNN(num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(3):
        total_loss = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/3, Loss: {total_loss / len(dataloader):.4f}")

    print("Training complete!\n")


# ==============================================================================
# Example 2: K-mer Features with MLP Classifier
# ==============================================================================


class KmerClassifierMLP(nn.Module):
    """MLP classifier using k-mer features."""

    def __init__(self, num_kmers, num_classes=2):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(num_kmers, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.layers(x)


def train_kmer_classifier():
    """Train an MLP classifier on k-mer features."""

    if not BINDINGS_AVAILABLE:
        print("Skipping k-mer training - bindings not available")
        return

    print("=" * 70)
    print("Example 2: K-mer Features with MLP Classifier")
    print("=" * 70)

    # Create k-mer encoder
    encoder = dbp.KmerEncoder(k=5, canonical=True, encoding_type="dna")

    # Synthetic data
    sequences = [b"ACGTACGT" * 20 for _ in range(50)] + [
        b"TTGGCCAA" * 20 for _ in range(50)
    ]
    labels = [0] * 50 + [1] * 50

    # Encode all sequences
    encoded_batch = encoder.encode_batch([seq for seq in sequences])

    # Convert to tensors
    X = torch.from_numpy(encoded_batch)
    y = torch.tensor(labels, dtype=torch.long)

    # Create model (4^5 = 1024 possible 5-mers)
    model = KmerClassifierMLP(num_kmers=1024, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    model.train()
    for epoch in range(5):
        optimizer.zero_grad()

        logits = model(X)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/5, Loss: {loss.item():.4f}")

    print("Training complete!\n")


# ==============================================================================
# Example 3: RNN with Integer Encoding
# ==============================================================================


class DNAClassifierLSTM(nn.Module):
    """LSTM classifier using integer encoding."""

    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=64, num_classes=2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x shape: [batch, seq_len] (integer encoded)

        # Embed
        embedded = self.embedding(x)  # [batch, seq_len, embedding_dim]

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(embedded)

        # Use final hidden states (forward + backward)
        h_forward = h_n[0]  # [batch, hidden_dim]
        h_backward = h_n[1]  # [batch, hidden_dim]
        h_concat = torch.cat([h_forward, h_backward], dim=1)

        # Classify
        logits = self.classifier(h_concat)

        return logits


def train_lstm_classifier():
    """Train an LSTM classifier with integer encoding."""

    if not BINDINGS_AVAILABLE:
        print("Skipping LSTM training - bindings not available")
        return

    print("=" * 70)
    print("Example 3: LSTM with Integer Encoding")
    print("=" * 70)

    # Create integer encoder
    encoder = dbp.IntegerEncoder("dna")

    # Synthetic data
    sequences = [b"ACGTACGT" * 10 for _ in range(50)] + [
        b"TTGGCCAA" * 10 for _ in range(50)
    ]
    labels = [0] * 50 + [1] * 50

    # Encode batch
    encoded_batch = encoder.encode_batch([seq for seq in sequences])

    # Convert to tensors
    # Remap: -1 (padding) → 0, A/C/G/T (0-3) → 1-4
    input_ids = torch.from_numpy(encoded_batch).long() + 1
    input_ids[input_ids == 0] = 0  # Keep padding as 0

    y = torch.tensor(labels, dtype=torch.long)

    # Create model (vocab: PAD=0, A=1, C=2, G=3, T=4)
    model = DNAClassifierLSTM(vocab_size=5, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    model.train()
    for epoch in range(5):
        optimizer.zero_grad()

        logits = model(input_ids)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/5, Loss: {loss.item():.4f}")

    print("Training complete!\n")


# ==============================================================================
# Main
# ==============================================================================


def main():
    print("\n" + "=" * 70)
    print("DeepBioP + PyTorch Integration Examples")
    print("=" * 70 + "\n")

    if not BINDINGS_AVAILABLE:
        print("⚠️  Python bindings not yet available.")
        print("These examples will work after implementing tasks T028-T032.\n")
        print("Expected API usage patterns are demonstrated in the code.")
        return

    # Run examples
    train_cnn_classifier()
    train_kmer_classifier()
    train_lstm_classifier()

    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
