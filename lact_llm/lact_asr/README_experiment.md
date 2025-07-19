# ASR Chunk TTT vs. Token TTT Experiment

This experiment compares:
- **Baseline ASR** (no test-time training)
- **Chunk Test-Time Training (LaCT)**
- **Token Test-Time Training (traditional TTT)**

on the LibriSpeech dataset using a simple encoder-decoder architecture and log-mel features.

## Usage
this is a simple test.

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Download LibriSpeech data:**
   The first run will download the `test-clean` split to `./data`.

3. **Run the experiment:**
   ```bash
   python run_asr_experiment.py
   ```
   This will print WER, CER, and hardware metrics for a few samples.

4. **Plot results:**
   After running, use the plotting script to visualize WER/CER across samples:
   ```bash
   python plot_results.py
   ```

## Files
- `run_asr_experiment.py`: Runs baseline, chunk TTT, and token TTT on LibriSpeech, prints and saves results.
- `plot_results.py`: Plots WER/CER for each method.
- `utils.py`: Feature extraction, metrics, hardware logging.
- `data.py`: LibriSpeech loader.
- `modeling_asr_lact.py`: Chunk TTT model.
- `modeling_asr_token_ttt.py`: Token TTT model.

## Output
- Prints WER/CER for each sample and method.
- Saves results to `results_asr_experiment.csv` for plotting.
- Plots WER/CER comparison across methods.

## Citation
If you use this code, please cite the main LaCT/TTT paper and this repository. 