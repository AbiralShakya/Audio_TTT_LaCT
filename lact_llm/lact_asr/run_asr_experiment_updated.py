import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from lact_llm.lact_asr.data import get_librispeech_loader
from lact_llm.lact_asr.utils import compute_wer, compute_cer, log_hardware_metrics
from lact_llm.lact_asr.modeling_asr_wav2vec2_lact import Wav2Vec2WithLaCT, BaselineWav2Vec2
import pandas as pd
import time
import gc

def main():
    """
    Run ASR experiment comparing:
    1. Baseline Wav2Vec2 (no TTT)
    2. Wav2Vec2 with LaCT (chunk-based test-time training)
    """
    
    # Configuration
    wav2vec2_model_id = "facebook/wav2vec2-base-960h"
    num_samples = 10  # Number of samples to evaluate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    print(f"Loading Wav2Vec2 model: {wav2vec2_model_id}")
    
    # Load Wav2Vec2 processor and base model
    processor = Wav2Vec2Processor.from_pretrained(wav2vec2_model_id)
    base_wav2vec2 = Wav2Vec2ForCTC.from_pretrained(wav2vec2_model_id)
    base_wav2vec2.eval()
    
    # Create models
    baseline_model = BaselineWav2Vec2(base_wav2vec2)
    baseline_model.to(device)
    baseline_model.eval()
    
    # Create Wav2Vec2 with LaCT model
    lact_model = Wav2Vec2WithLaCT(
        wav2vec2_model=base_wav2vec2,
        num_lact_layers=2,
        lact_hidden_size=768,
        num_lact_heads=4,
        lact_chunk_size=512,
        lact_lr_scale=0.01,
        use_momentum=True
    )
    lact_model.to(device)
    lact_model.eval()
    
    # Load LibriSpeech data
    print("Loading LibriSpeech data...")
    loader = get_librispeech_loader("./data", url="test-clean", batch_size=1)
    
    # Results storage
    results = []
    
    print(f"Starting evaluation on {num_samples} samples...")
    print("-" * 80)
    
    for i, (waveform, sample_rate, transcript) in enumerate(loader):
        if i >= num_samples:
            break
            
        print(f"Processing sample {i+1}/{num_samples}")
        
        # Prepare input
        input_wav = waveform[0].squeeze().numpy()
        reference = transcript[0].lower().strip()
        
        # Process with Wav2Vec2 processor
        inputs = processor(
            input_wav, 
            sampling_rate=sample_rate[0], 
            return_tensors="pt", 
            padding=True
        )
        input_values = inputs["input_values"].to(device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # Baseline Wav2Vec2 (no TTT)
        print("  Running baseline Wav2Vec2...")
        start_time = time.time()
        baseline_model.reset_fast_weights() if hasattr(baseline_model, 'reset_fast_weights') else None
        
        with torch.no_grad():
            baseline_outputs = baseline_model(input_values, attention_mask=attention_mask)
            baseline_logits = baseline_outputs.logits if hasattr(baseline_outputs, 'logits') else baseline_outputs['logits']
            baseline_pred_ids = torch.argmax(baseline_logits, dim=-1)
            baseline_hypothesis = processor.batch_decode(baseline_pred_ids)[0].lower().strip()
        
        baseline_time = time.time() - start_time
        
        # Wav2Vec2 with LaCT (chunk-based TTT)
        print("  Running Wav2Vec2 with LaCT...")
        start_time = time.time()
        lact_model.reset_fast_weights()  # Reset fast weights for new sequence
        
        with torch.no_grad():
            lact_outputs = lact_model(input_values, attention_mask=attention_mask)
            lact_logits = lact_outputs['logits']
            lact_pred_ids = torch.argmax(lact_logits, dim=-1)
            lact_hypothesis = processor.batch_decode(lact_pred_ids)[0].lower().strip()
        
        lact_time = time.time() - start_time
        
        # Compute metrics
        baseline_wer = compute_wer(reference, baseline_hypothesis)
        baseline_cer = compute_cer(reference, baseline_hypothesis)
        lact_wer = compute_wer(reference, lact_hypothesis)
        lact_cer = compute_cer(reference, lact_hypothesis)
        
        # Log hardware metrics
        gpu_memory = log_hardware_metrics()
        
        # Print results
        print(f"  Reference: {reference}")
        print(f"  Baseline:  {baseline_hypothesis}")
        print(f"  LaCT:      {lact_hypothesis}")
        print(f"  WER - Baseline: {baseline_wer:.3f}, LaCT: {lact_wer:.3f}")
        print(f"  CER - Baseline: {baseline_cer:.3f}, LaCT: {lact_cer:.3f}")
        print(f"  Time - Baseline: {baseline_time:.3f}s, LaCT: {lact_time:.3f}s")
        print(f"  GPU Memory: {gpu_memory:.1f} MB")
        print("-" * 80)
        
        # Store results
        results.append({
            'sample_id': i + 1,
            'reference': reference,
            'baseline_hypothesis': baseline_hypothesis,
            'lact_hypothesis': lact_hypothesis,
            'baseline_wer': baseline_wer,
            'lact_wer': lact_wer,
            'baseline_cer': baseline_cer,
            'lact_cer': lact_cer,
            'baseline_time': baseline_time,
            'lact_time': lact_time,
            'gpu_memory_mb': gpu_memory
        })
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv("results_wav2vec2_lact_comparison.csv", index=False)
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    avg_baseline_wer = df['baseline_wer'].mean()
    avg_lact_wer = df['lact_wer'].mean()
    avg_baseline_cer = df['baseline_cer'].mean()
    avg_lact_cer = df['lact_cer'].mean()
    avg_baseline_time = df['baseline_time'].mean()
    avg_lact_time = df['lact_time'].mean()
    
    print(f"Average WER - Baseline: {avg_baseline_wer:.3f}, LaCT: {avg_lact_wer:.3f}")
    print(f"Average CER - Baseline: {avg_baseline_cer:.3f}, LaCT: {avg_lact_cer:.3f}")
    print(f"Average Time - Baseline: {avg_baseline_time:.3f}s, LaCT: {avg_lact_time:.3f}s")
    
    # Compute improvement
    wer_improvement = avg_baseline_wer - avg_lact_wer
    cer_improvement = avg_baseline_cer - avg_lact_cer
    time_overhead = avg_lact_time - avg_baseline_time
    
    print(f"\nLaCT vs Baseline:")
    print(f"  WER Improvement: {wer_improvement:+.3f} ({wer_improvement/avg_baseline_wer*100:+.1f}%)")
    print(f"  CER Improvement: {cer_improvement:+.3f} ({cer_improvement/avg_baseline_cer*100:+.1f}%)")
    print(f"  Time Overhead: {time_overhead:+.3f}s ({time_overhead/avg_baseline_time*100:+.1f}%)")
    
    print(f"\nResults saved to: results_wav2vec2_lact_comparison.csv")

if __name__ == "__main__":
    main() 