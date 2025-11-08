"""
Integration tests for PyTorch and HuggingFace Transformers compatibility.

These tests verify that DeepBioP encodings work seamlessly with major ML frameworks.

Run with: pytest tests/test_ml_framework_compat.py
"""

import numpy as np
import pytest

# These imports will work once we implement the Python bindings
# pytest.importorskip ensures tests are skipped if dependencies are missing
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


# These will be implemented in T028-T032
# For now, these are the expected interfaces
@pytest.fixture
def sample_dna_sequences():
    """Sample DNA sequences for testing."""
    return [
        b"ACGTACGTACGTACGT",  # 16 bp
        b"AAACCCTTTGGG",  # 12 bp
        b"ACGTACGT",  # 8 bp
    ]


class TestOneHotPyTorchCompatibility:
    """Test one-hot encoding with PyTorch."""

    @pytest.mark.skip(reason="Waiting for Python bindings implementation (T028-T029)")
    def test_onehot_numpy_array_properties(self):
        """Verify one-hot output is a proper NumPy array."""
        import deepbiop as dbp

        encoder = dbp.OneHotEncoder("dna", "skip")
        encoded = encoder.encode(b"ACGT")

        # Check NumPy properties
        assert isinstance(encoded, np.ndarray)
        assert encoded.dtype == np.float32
        assert encoded.flags["C_CONTIGUOUS"]
        assert encoded.shape == (4, 4)

    @pytest.mark.skip(reason="Waiting for Python bindings implementation (T028-T029)")
    def test_onehot_pytorch_tensor_conversion(self):
        """Verify zero-copy conversion to PyTorch tensor."""
        import deepbiop as dbp

        encoder = dbp.OneHotEncoder("dna", "skip")
        encoded = encoder.encode(b"ACGTACGT")

        # Convert to PyTorch tensor
        tensor = torch.from_numpy(encoded)

        # Verify properties
        assert tensor.dtype == torch.float32
        assert tensor.shape == (8, 4)
        assert tensor.is_contiguous()

        # Verify zero-copy (shares memory)
        assert tensor.data_ptr() == encoded.ctypes.data

    @pytest.mark.skip(reason="Waiting for Python bindings implementation (T028-T029)")
    def test_onehot_batch_pytorch(self, sample_dna_sequences):
        """Test batch encoding with PyTorch."""
        import deepbiop as dbp

        encoder = dbp.OneHotEncoder("dna", "skip")
        batch = encoder.encode_batch(sample_dna_sequences)

        # Convert to tensor
        tensor = torch.from_numpy(batch)

        # Verify shape: [batch_size, max_length, 4]
        assert tensor.shape == (3, 16, 4)  # Padded to longest (16bp)
        assert tensor.dtype == torch.float32

        # Verify padding is zeros
        assert torch.all(tensor[1, 12:, :] == 0)  # Sequence 2 padded after position 12
        assert torch.all(tensor[2, 8:, :] == 0)  # Sequence 3 padded after position 8

    @pytest.mark.skip(reason="Waiting for Python bindings implementation (T028-T029)")
    def test_onehot_cnn_forward_pass(self, sample_dna_sequences):
        """Test one-hot encoding through a CNN model."""
        import torch.nn as nn

        import deepbiop as dbp

        encoder = dbp.OneHotEncoder("dna", "skip")
        batch = encoder.encode_batch(sample_dna_sequences)

        # Convert to tensor and permute for Conv1d: [batch, channels, length]
        X = torch.from_numpy(batch).permute(0, 2, 1)

        # Simple CNN
        model = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 2),
        )

        # Forward pass
        logits = model(X)

        assert logits.shape == (3, 2)  # [batch_size, num_classes]
        assert not torch.isnan(logits).any()

    @pytest.mark.skip(reason="Waiting for Python bindings implementation (T028-T029)")
    def test_onehot_gradient_flow(self):
        """Verify gradients flow correctly through encodings."""
        import deepbiop as dbp

        encoder = dbp.OneHotEncoder("dna", "skip")
        encoded = encoder.encode(b"ACGTACGT")

        tensor = torch.from_numpy(encoded)
        tensor.requires_grad_(True)

        # Simple operation
        output = tensor.sum()
        output.backward()

        # Verify gradients exist
        assert tensor.grad is not None
        assert tensor.grad.shape == tensor.shape


class TestKmerPyTorchCompatibility:
    """Test k-mer encoding with PyTorch."""

    @pytest.mark.skip(reason="Waiting for Python bindings implementation (T028-T030)")
    def test_kmer_pytorch_conversion(self):
        """Test k-mer encoding to PyTorch tensor."""
        import deepbiop as dbp

        encoder = dbp.KmerEncoder(k=3, canonical=False, encoding_type="dna")
        encoded = encoder.encode(b"ACGTACGT")

        tensor = torch.from_numpy(encoded)

        # For k=3, DNA: 4^3 = 64 possible k-mers
        assert tensor.shape == (64,)
        assert tensor.dtype == torch.float32

    @pytest.mark.skip(reason="Waiting for Python bindings implementation (T028-T030)")
    def test_kmer_mlp_classifier(self, sample_dna_sequences):
        """Test k-mer encoding through MLP classifier."""
        import torch.nn as nn

        import deepbiop as dbp

        encoder = dbp.KmerEncoder(k=5, canonical=True, encoding_type="dna")
        batch = encoder.encode_batch(sample_dna_sequences)

        X = torch.from_numpy(batch)

        # MLP classifier
        model = nn.Sequential(
            nn.Linear(1024, 256),  # 4^5 = 1024 for k=5
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

        logits = model(X)
        assert logits.shape == (3, 2)


class TestIntegerEncodingPyTorchCompatibility:
    """Test integer encoding with PyTorch."""

    @pytest.mark.skip(reason="Waiting for Python bindings implementation (T028-T030)")
    def test_integer_pytorch_conversion(self):
        """Test integer encoding to PyTorch tensor."""
        import deepbiop as dbp

        encoder = dbp.IntegerEncoder("dna")
        encoded = encoder.encode(b"ACGT")

        # Convert to long tensor for embedding layers
        tensor = torch.from_numpy(encoded).long()

        assert tensor.shape == (4,)
        assert tensor.dtype == torch.int64
        assert torch.all(tensor >= 0)
        assert torch.all(tensor <= 3)  # DNA: A=0, C=1, G=2, T=3

    @pytest.mark.skip(reason="Waiting for Python bindings implementation (T028-T030)")
    def test_integer_embedding_layer(self):
        """Test integer encoding through embedding layer."""
        import torch.nn as nn

        import deepbiop as dbp

        encoder = dbp.IntegerEncoder("dna")
        encoded = encoder.encode(b"ACGTACGT")

        tensor = torch.from_numpy(encoded).long()

        # Embedding layer
        embedding = nn.Embedding(num_embeddings=4, embedding_dim=16)
        embedded = embedding(tensor)

        assert embedded.shape == (8, 16)  # [seq_len, embedding_dim]

    @pytest.mark.skip(reason="Waiting for Python bindings implementation (T028-T030)")
    def test_integer_batch_with_attention_mask(self, sample_dna_sequences):
        """Test creating attention masks from integer encoding."""
        import deepbiop as dbp

        encoder = dbp.IntegerEncoder("dna")
        batch = encoder.encode_batch(sample_dna_sequences)

        input_ids = torch.from_numpy(batch).long()
        attention_mask = (input_ids != -1).long()

        # Verify shapes
        assert input_ids.shape[0] == 3  # batch size
        assert attention_mask.shape == input_ids.shape

        # Verify mask correctness
        assert attention_mask[0].sum() == 16  # First sequence: 16 bp
        assert attention_mask[1].sum() == 12  # Second sequence: 12 bp
        assert attention_mask[2].sum() == 8  # Third sequence: 8 bp


class TestHuggingFaceTransformersCompatibility:
    """Test integration with HuggingFace Transformers."""

    @pytest.mark.skip(reason="Waiting for Python bindings implementation (T028-T030)")
    def test_bert_model_forward(self, sample_dna_sequences):
        """Test integer encoding with BERT model."""
        from transformers import BertConfig, BertModel

        import deepbiop as dbp

        encoder = dbp.IntegerEncoder("dna")
        batch = encoder.encode_batch(sample_dna_sequences)

        # Prepare transformer inputs
        # Need to remap: DNA tokens (0-3) → vocab indices (3-6)
        # Reserve 0-2 for [PAD], [CLS], [SEP]
        input_ids = torch.from_numpy(batch).long()
        input_ids = input_ids + 3  # Offset DNA tokens
        input_ids[input_ids == 2] = 0  # Remap padding (-1 + 3 = 2) to PAD token

        attention_mask = (input_ids != 0).long()

        # Configure small BERT for testing
        config = BertConfig(
            vocab_size=7,  # PAD, CLS, SEP, A, C, G, T
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=32,
        )

        model = BertModel(config)
        model.eval()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Verify output
        assert outputs.last_hidden_state.shape == (
            3,
            16,
            64,
        )  # [batch, seq_len, hidden]
        assert not torch.isnan(outputs.last_hidden_state).any()

    @pytest.mark.skip(reason="Waiting for Python bindings implementation (T028-T030)")
    def test_bert_for_classification(self, sample_dna_sequences):
        """Test end-to-end classification with BERT."""
        from transformers import BertConfig, BertForSequenceClassification

        import deepbiop as dbp

        encoder = dbp.IntegerEncoder("dna")
        batch = encoder.encode_batch(sample_dna_sequences)

        # Prepare inputs (with token remapping)
        input_ids = torch.from_numpy(batch).long() + 3
        input_ids[input_ids == 2] = 0
        attention_mask = (input_ids != 0).long()

        # Classification model
        config = BertConfig(
            vocab_size=7,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
        )
        model = BertForSequenceClassification(config, num_labels=2)
        model.eval()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Verify logits
        assert outputs.logits.shape == (3, 2)  # [batch, num_labels]


class TestDataTypeConversions:
    """Test various data type conversion scenarios."""

    @pytest.mark.skip(reason="Waiting for Python bindings implementation (T028-T030)")
    def test_float32_to_float16(self):
        """Test conversion to half precision."""
        import deepbiop as dbp

        encoder = dbp.OneHotEncoder("dna", "skip")
        encoded = encoder.encode(b"ACGT")

        tensor = torch.from_numpy(encoded)
        tensor_fp16 = tensor.half()

        assert tensor_fp16.dtype == torch.float16

    @pytest.mark.skip(reason="Waiting for Python bindings implementation (T028-T030)")
    def test_cpu_to_gpu_transfer(self):
        """Test GPU transfer if CUDA is available."""
        import deepbiop as dbp

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        encoder = dbp.OneHotEncoder("dna", "skip")
        encoded = encoder.encode(b"ACGTACGT")

        tensor_cpu = torch.from_numpy(encoded)
        tensor_gpu = tensor_cpu.cuda()

        assert tensor_gpu.device.type == "cuda"
        assert torch.all(tensor_cpu == tensor_gpu.cpu())


def test_numpy_compatibility_placeholder():
    """
    Placeholder test to ensure test file is valid.

    Remove this once actual bindings are implemented.
    """
    # This test ensures the file can be collected by pytest
    assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
