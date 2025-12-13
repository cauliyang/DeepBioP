#!/usr/bin/env python3
"""
Example: Using DeepBioP with HuggingFace Transformers for DNA Sequences

This example demonstrates how to use DeepBioP integer encoding with
HuggingFace Transformers (BERT, etc.) for DNA sequence tasks.

Requirements:
    pip install deepbiop transformers torch numpy

Usage:
    python examples/transformers_integration.py
"""

try:
    import torch
    from transformers import BertConfig, BertModel, BertForSequenceClassification
    from transformers import TrainingArguments, Trainer
except ImportError:
    print("Required libraries not installed.")
    print("Install with: pip install torch transformers")
    exit(1)

# Will work once Python bindings are implemented
try:
    import deepbiop as dbp

    BINDINGS_AVAILABLE = True
except ImportError:
    print("DeepBioP Python bindings not yet available.")
    print("This example will work after completing tasks T028-T032.")
    BINDINGS_AVAILABLE = False


# ==============================================================================
# Utility Functions for Token Remapping
# ==============================================================================


def prepare_transformer_inputs(encoded_batch, special_tokens_offset=5):
    """
    Prepare DeepBioP integer encoding for transformer input.

    DeepBioP integer encoding:
        - DNA: A=0, C=1, G=2, T=3
        - Padding: -1

    Transformer vocabulary:
        - 0: [PAD]
        - 1: [CLS]
        - 2: [SEP]
        - 3: [MASK]
        - 4: [UNK]
        - 5: A
        - 6: C
        - 7: G
        - 8: T

    Args:
        encoded_batch: NumPy array from IntegerEncoder [batch, seq_len]
        special_tokens_offset: Offset for DNA tokens (default 5)

    Returns:
        input_ids: Tensor with remapped token IDs
        attention_mask: Binary mask (1 for real tokens, 0 for padding)
    """
    # Convert to tensor
    input_ids = torch.from_numpy(encoded_batch).long()

    # Remap: DNA tokens (0-3) → vocab indices (5-8)
    input_ids = input_ids + special_tokens_offset

    # Remap padding: (-1 + 5 = 4) → 0 ([PAD])
    input_ids[input_ids == 4] = 0

    # Create attention mask
    attention_mask = (input_ids != 0).long()

    return input_ids, attention_mask


def add_special_tokens(input_ids, attention_mask, cls_token_id=1, sep_token_id=2):
    """
    Add [CLS] and [SEP] tokens to sequences.

    Args:
        input_ids: Tensor [batch, seq_len]
        attention_mask: Tensor [batch, seq_len]
        cls_token_id: ID for [CLS] token
        sep_token_id: ID for [SEP] token

    Returns:
        Modified input_ids and attention_mask with special tokens
    """
    batch_size, seq_len = input_ids.shape

    # Create new tensors with space for [CLS] and [SEP]
    new_input_ids = torch.zeros(batch_size, seq_len + 2, dtype=torch.long)
    new_attention_mask = torch.zeros(batch_size, seq_len + 2, dtype=torch.long)

    # Add [CLS] at the beginning
    new_input_ids[:, 0] = cls_token_id
    new_attention_mask[:, 0] = 1

    # Copy original tokens
    new_input_ids[:, 1 : seq_len + 1] = input_ids
    new_attention_mask[:, 1 : seq_len + 1] = attention_mask

    # Add [SEP] at the end of each sequence
    for i in range(batch_size):
        # Find last non-padding token
        last_token_pos = attention_mask[i].sum().item()
        new_input_ids[i, last_token_pos + 1] = sep_token_id
        new_attention_mask[i, last_token_pos + 1] = 1

    return new_input_ids, new_attention_mask


# ==============================================================================
# Example 1: BERT for DNA Sequence Classification
# ==============================================================================


def create_dna_bert_model(num_labels=2, max_length=512):
    """Create a BERT model configured for DNA sequences."""

    config = BertConfig(
        vocab_size=9,  # PAD, CLS, SEP, MASK, UNK, A, C, G, T
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=1024,
        max_position_embeddings=max_length,
        type_vocab_size=1,  # No segment embeddings needed
    )

    model = BertForSequenceClassification(config, num_labels=num_labels)

    return model


def example_bert_classification():
    """Example: DNA sequence classification with BERT."""

    if not BINDINGS_AVAILABLE:
        print("Skipping BERT example - bindings not available")
        return

    print("=" * 70)
    print("Example 1: BERT for DNA Sequence Classification")
    print("=" * 70)

    # Create encoder
    encoder = dbp.IntegerEncoder("dna")

    # Sample sequences (synthetic data)
    sequences = [
        b"ACGTACGTACGTACGT" * 5,  # 80 bp, class 0
        b"TTGGCCAATTGGCCAA" * 5,  # 80 bp, class 0
        b"AAAACCCCGGGGTTTT" * 5,  # 80 bp, class 1
        b"CGCGCGCGCGCGCGCG" * 5,  # 80 bp, class 1
    ] * 10  # 40 samples

    labels = torch.tensor(([0, 0, 1, 1] * 10), dtype=torch.long)

    # Encode sequences
    encoded_batch = encoder.encode_batch(sequences)

    # Prepare transformer inputs
    input_ids, attention_mask = prepare_transformer_inputs(encoded_batch)
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")

    # Create model
    model = create_dna_bert_model(num_labels=2, max_length=128)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    print(f"Logits shape: {logits.shape}")
    print(f"Sample predictions: {logits[:4].argmax(dim=1)}")
    print(f"True labels: {labels[:4]}")
    print()


# ==============================================================================
# Example 2: Pre-training BERT on DNA Sequences (Masked Language Modeling)
# ==============================================================================


def example_bert_pretraining():
    """Example: Pre-training BERT with masked language modeling."""

    if not BINDINGS_AVAILABLE:
        print("Skipping pre-training example - bindings not available")
        return

    print("=" * 70)
    print("Example 2: BERT Pre-training (Masked Language Modeling)")
    print("=" * 70)

    from transformers import BertForMaskedLM

    # Create encoder
    encoder = dbp.IntegerEncoder("dna")

    # Sample sequences
    sequences = [b"ACGTACGT" * 20 for _ in range(100)]  # 160 bp each

    # Encode
    encoded_batch = encoder.encode_batch(sequences)
    input_ids, attention_mask = prepare_transformer_inputs(encoded_batch)

    # Create model for MLM
    config = BertConfig(
        vocab_size=9, hidden_size=128, num_hidden_layers=4, num_attention_heads=4
    )

    BertForMaskedLM(config)

    # Create data collator for MLM (15% masking)
    # Note: This would need a custom tokenizer or preprocessing
    # For demonstration purposes only

    print("Model ready for pre-training")
    print(f"Vocabulary size: {config.vocab_size}")
    print(f"Sample input shape: {input_ids.shape}")
    print()


# ==============================================================================
# Example 3: Sequence Embedding Extraction
# ==============================================================================


def example_sequence_embeddings():
    """Example: Extract sequence embeddings from BERT."""

    if not BINDINGS_AVAILABLE:
        print("Skipping embedding example - bindings not available")
        return

    print("=" * 70)
    print("Example 3: Extract Sequence Embeddings")
    print("=" * 70)

    # Create encoder
    encoder = dbp.IntegerEncoder("dna")

    # Sample sequences
    sequences = [
        b"ACGTACGTACGTACGT",
        b"TTGGCCAATTGGCCAA",
        b"AAAACCCCGGGGTTTT",
    ]

    # Encode
    encoded_batch = encoder.encode_batch(sequences)
    input_ids, attention_mask = prepare_transformer_inputs(encoded_batch)

    # Create BERT model (without classification head)
    config = BertConfig(vocab_size=9, hidden_size=256, num_hidden_layers=4)
    model = BertModel(config)
    model.eval()

    # Extract embeddings
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Get [CLS] token embedding (sequence-level)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]

        # Get mean pooled embedding
        sequence_lengths = attention_mask.sum(dim=1, keepdim=True)
        mean_embeddings = (
            outputs.last_hidden_state * attention_mask.unsqueeze(-1)
        ).sum(dim=1) / sequence_lengths

    print(f"[CLS] embeddings shape: {cls_embeddings.shape}")
    print(f"Mean pooled embeddings shape: {mean_embeddings.shape}")

    # These embeddings can be used for:
    # - Similarity search
    # - Clustering
    # - Downstream classification
    # - Retrieval tasks

    print()


# ==============================================================================
# Example 4: Fine-tuning with HuggingFace Trainer
# ==============================================================================


class DNASequenceDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for DNA sequences with transformer inputs."""

    def __init__(self, sequences, labels, encoder):
        self.sequences = sequences
        self.labels = labels
        self.encoder = encoder

        # Pre-encode all sequences
        encoded_batch = encoder.encode_batch(sequences)
        self.input_ids, self.attention_mask = prepare_transformer_inputs(encoded_batch)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def example_trainer_api():
    """Example: Fine-tuning with HuggingFace Trainer API."""

    if not BINDINGS_AVAILABLE:
        print("Skipping Trainer example - bindings not available")
        return

    print("=" * 70)
    print("Example 4: Fine-tuning with HuggingFace Trainer")
    print("=" * 70)

    # Create encoder
    encoder = dbp.IntegerEncoder("dna")

    # Synthetic dataset
    train_sequences = [b"ACGT" * 20 for _ in range(80)] + [
        b"TTGG" * 20 for _ in range(80)
    ]
    train_labels = [0] * 80 + [1] * 80

    val_sequences = [b"ACGT" * 20 for _ in range(10)] + [
        b"TTGG" * 20 for _ in range(10)
    ]
    val_labels = [0] * 10 + [1] * 10

    # Create datasets
    train_dataset = DNASequenceDataset(train_sequences, train_labels, encoder)
    val_dataset = DNASequenceDataset(val_sequences, val_labels, encoder)

    # Create model
    model = create_dna_bert_model(num_labels=2)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        learning_rate=5e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
    )

    # Create Trainer
    Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train
    print("Starting training...")
    # trainer.train()  # Commented out for demonstration

    print("Training would start here with trainer.train()")
    print()


# ==============================================================================
# Main
# ==============================================================================


def main():
    print("\n" + "=" * 70)
    print("DeepBioP + HuggingFace Transformers Integration Examples")
    print("=" * 70 + "\n")

    if not BINDINGS_AVAILABLE:
        print("⚠️  Python bindings not yet available.")
        print("These examples will work after implementing tasks T028-T032.\n")
        print("Expected API usage patterns are demonstrated in the code.")
        return

    # Run examples
    example_bert_classification()
    example_bert_pretraining()
    example_sequence_embeddings()
    example_trainer_api()

    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
