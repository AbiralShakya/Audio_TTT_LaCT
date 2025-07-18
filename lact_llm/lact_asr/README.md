# LaCT-ASR: Chunk Test-Time Training for ASR

This submodule explores **chunk-based Test-Time Training (LaCT)** versus **traditional token-based TTT** for Automatic Speech Recognition (ASR).

## Research Goal
- **Compare**: Chunk TTT (LaCT) vs. Token TTT for ASR.
- **Task**: Speech-to-text (ASR) on standard datasets (e.g., LibriSpeech).
- **Hypothesis**: Chunk TTT will improve hardware efficiency and long-context adaptation compared to token TTT, with competitive or better accuracy.

## Architecture
- **Base Model**: Transformer-based ASR (e.g., Wav2Vec2, or custom encoder-decoder from `lact_llm`).
- **TTT Variants**:
  - **Chunk TTT (LaCT)**: Adapt fast weights every large chunk (e.g., 2K tokens).
  - **Token TTT**: Adapt fast weights every token (traditional approach).

## Experiments
- **Metrics**: WER (Word Error Rate), CER (Character Error Rate), hardware metrics (throughput, memory).
- **Compare**:
  - Baseline (no TTT)
  - Token TTT
  - Chunk TTT (LaCT)

## Structure
- `modeling_asr_lact.py`: LaCT/Chunk TTT ASR model
- `modeling_asr_token_ttt.py`: Token TTT ASR model
- `run_asr_experiment.py`: Training/inference/metrics script
- `metrics.py`: WER, CER, hardware logging

## Status
- [ ] Model adaptation
- [ ] Experiment scripts
- [ ] Evaluation

---

**Contact:** For research inquiries, see main repo or contact maintainers. 