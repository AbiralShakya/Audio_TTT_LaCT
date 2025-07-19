# Wav2Vec2 with LaCT Integration for ASR

This module implements **chunk-based test-time training (LaCT)** on top of Hugging Face's Wav2Vec2 model for Automatic Speech Recognition (ASR), following the methodology described in "Test Time Training Done Right".

## Overview

The integration combines:
- **Hugging Face Wav2Vec2**: A state-of-the-art ASR model
- **LaCT (Large Chunk Test-Time Training)**: Chunk-based fast weight adaptation during inference
- **Real ASR Evaluation**: Using LibriSpeech dataset with WER/CER metrics

## Key Components

### 1. LaCT Layer (`modeling_asr_wav2vec2_lact.py`)

```python
class LaCTLayer(nn.Module):
    """
    Implements chunk-based test-time training with fast weight updates
    """
    def __init__(self, hidden_size=768, num_lact_heads=4, chunk_size=512, ...):
        # Fast weight matrices (w0, w1, w2) for SwiGLU-style processing
        # Momentum buffers for stable updates
        # Learnable learning rate parameters
```

**Key Features:**
- **Chunk-based processing**: Splits input into large chunks (e.g., 512 tokens)
- **Fast weight updates**: Updates w0, w1, w2 matrices during inference
- **Momentum-based optimization**: Stable weight updates with momentum
- **SwiGLU activation**: Modern activation function for better performance

### 2. Wav2Vec2 with LaCT Wrapper

```python
class Wav2Vec2WithLaCT(nn.Module):
    """
    Wraps Wav2Vec2 with LaCT layers for chunk-based test-time training
    """
    def __init__(self, wav2vec2_model, num_lact_layers=2, ...):
        # Adds LaCT layers on top of Wav2Vec2
        # Handles input/output projections
        # Manages fast weight reset between sequences
```

**Architecture:**
```
Input Audio → Wav2Vec2 Encoder → LaCT Layer 1 → LaCT Layer 2 → Output Projection → Logits
```

### 3. Experiment Pipeline

The experiment compares:
- **Baseline**: Standard Wav2Vec2 without TTT
- **LaCT**: Wav2Vec2 with chunk-based test-time training

## Usage

### 1. Run the Experiment

```bash
cd lact_llm/lact_asr
python run_asr_experiment_updated.py
```

This will:
- Load Wav2Vec2 model from Hugging Face
- Create baseline and LaCT variants
- Evaluate on LibriSpeech test-clean
- Save results to `results_wav2vec2_lact_comparison.csv`

### 2. Plot Results

```bash
python plot_wav2vec2_lact_results.py
```

Generates comparison plots:
- WER/CER per sample
- Inference time comparison
- Summary statistics

## Configuration

### LaCT Parameters

```python
lact_config = {
    'num_lact_layers': 2,        # Number of LaCT layers
    'lact_hidden_size': 768,     # Hidden dimension
    'num_lact_heads': 4,         # Number of fast weight heads
    'lact_chunk_size': 512,      # Chunk size for TTT
    'lact_lr_scale': 0.01,       # Learning rate scale
    'use_momentum': True,        # Use momentum for updates
    'momentum': 0.9              # Momentum value
}
```

### Model Variants

1. **Baseline Wav2Vec2**: `facebook/wav2vec2-base-960h`
2. **Wav2Vec2 + LaCT**: Same base model with LaCT layers

## How LaCT Works

### 1. Chunk-based Processing
```
Input: [B, T, D] → Split into chunks of size 512
Chunk 1: [B, 512, D] → Process with fast weights
Chunk 2: [B, 512, D] → Update fast weights → Process
...
```

### 2. Fast Weight Updates
For each chunk:
1. **Compute update signal**: Attention-like mechanism
2. **Update fast weights**: w0, w1, w2 matrices
3. **Apply fast weights**: SwiGLU-style processing
4. **Momentum update**: Stabilize weight changes

### 3. Test-Time Training
- **No pre-training**: Fast weights start from small random values
- **Online adaptation**: Weights adapt during inference
- **Sequence-specific**: Reset weights for each new audio sequence

## Expected Results

### Performance Metrics
- **WER (Word Error Rate)**: Primary ASR metric
- **CER (Character Error Rate)**: Character-level accuracy
- **Inference Time**: Computational overhead of LaCT

### Typical Outcomes
- **Accuracy**: LaCT may improve WER/CER on longer sequences
- **Efficiency**: Small time overhead for chunk processing
- **Adaptation**: Fast weights adapt to sequence characteristics

## Integration with Existing Code

### From `lact_audio`
- **LaCT concept**: Chunk-based processing approach
- **Fast weight design**: SwiGLU-style architecture
- **Momentum optimization**: Stable weight updates

### From `lact_asr`
- **ASR evaluation**: WER/CER metrics
- **Data loading**: LibriSpeech integration
- **Hardware monitoring**: GPU memory and timing

## Research Context

This implementation follows the "Test Time Training Done Right" paper by:
1. **Using real ASR models**: Wav2Vec2 instead of toy models
2. **Chunk-based approach**: Large chunks rather than token-by-token
3. **Proper evaluation**: Real ASR metrics on standard dataset
4. **Hardware efficiency**: Monitoring computational overhead

## Files Structure

```
lact_llm/lact_asr/
├── modeling_asr_wav2vec2_lact.py    # Wav2Vec2 + LaCT implementation
├── run_asr_experiment_updated.py    # Main experiment script
├── plot_wav2vec2_lact_results.py    # Results visualization
├── data.py                          # LibriSpeech data loader
├── utils.py                         # Metrics and utilities
└── README_wav2vec2_lact_integration.md  # This file
```

## Dependencies

```bash
pip install torch torchaudio transformers pandas matplotlib seaborn jiwer
```

## Next Steps

1. **Hyperparameter tuning**: Optimize chunk size, learning rates
2. **Different base models**: Try larger Wav2Vec2 variants
3. **Training integration**: Combine with fine-tuning
4. **Real-time evaluation**: Test on streaming audio
5. **Ablation studies**: Analyze LaCT layer contributions

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch size or chunk size
2. **Import errors**: Install missing dependencies
3. **Data loading**: Ensure LibriSpeech is downloaded

### Performance Tips
1. **Chunk size**: Larger chunks = more context but slower
2. **LaCT layers**: More layers = better adaptation but slower
3. **Learning rate**: Higher lr = faster adaptation but instability

## Citation

If you use this implementation, please cite:
- "Test Time Training Done Right" paper
- Wav2Vec2 paper
- LaCT methodology 