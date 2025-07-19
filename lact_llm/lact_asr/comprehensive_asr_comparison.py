import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Model
from lact_llm.lact_asr.data import get_librispeech_loader
from lact_llm.lact_asr.utils import compute_wer, compute_cer, log_hardware_metrics
from lact_llm.lact_asr.modeling_asr_wav2vec2_lact import Wav2Vec2WithLaCT, BaselineWav2Vec2
import pandas as pd
import time
import gc
import numpy as np
from typing import Dict, List, Tuple, Any

class TokenBasedTTT(nn.Module):
    """
    Token-based Test-Time Training for ASR
    Updates fast weights after each token instead of chunks
    """
    def __init__(self, wav2vec2_model, num_ttt_layers=2, hidden_size=768, lr_scale=0.01):
        super().__init__()
        self.wav2vec2 = wav2vec2_model
        self.hidden_size = hidden_size
        self.lr_scale = lr_scale
        
        # Token-based fast weights (smaller than chunk-based)
        self.fast_weights = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.fast_bias = nn.Parameter(torch.zeros(hidden_size))
        
        # Projection layers
        wav2vec2_hidden_size = wav2vec2_model.config.hidden_size
        self.input_projection = nn.Linear(wav2vec2_hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, wav2vec2_model.config.vocab_size)
        self.norm = nn.LayerNorm(hidden_size)
        
    def update_fast_weights(self, hidden_states):
        """Update fast weights after each token"""
        # Simple gradient-like update based on current hidden state
        update = torch.matmul(hidden_states, hidden_states.transpose(-2, -1)) * self.lr_scale
        self.fast_weights.data += update.mean(dim=0)
        
    def forward(self, input_values, attention_mask=None, **kwargs):
        # Get Wav2Vec2 features
        wav2vec2_outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        
        hidden_states = wav2vec2_outputs.hidden_states[-1]  # [B, T, D]
        x = self.input_projection(hidden_states)
        
        # Token-by-token processing with TTT
        B, T, D = x.shape
        output = torch.zeros_like(x)
        
        for t in range(T):
            # Process current token
            token_hidden = x[:, t:t+1, :]  # [B, 1, D]
            
            # Apply fast weights
            token_out = torch.matmul(token_hidden, self.fast_weights) + self.fast_bias
            token_out = self.norm(token_out)
            
            output[:, t:t+1, :] = token_out
            
            # Update fast weights after each token
            if t < T - 1:  # Don't update after last token
                self.update_fast_weights(token_hidden)
        
        # Final projection
        logits = self.output_projection(output)
        
        return {
            'logits': logits,
            'hidden_states': wav2vec2_outputs.hidden_states,
            'attentions': wav2vec2_outputs.attentions
        }
    
    def reset_fast_weights(self):
        """Reset fast weights for new sequence"""
        self.fast_weights.data.zero_()
        self.fast_bias.data.zero_()

class FineTunedWav2Vec2(nn.Module):
    """
    Fine-tuned Wav2Vec2 model (simulated)
    Represents a model that has been fine-tuned on the target domain
    """
    def __init__(self, wav2vec2_model, fine_tune_strength=0.1):
        super().__init__()
        self.wav2vec2 = wav2vec2_model
        
        # Simulate fine-tuning by adding small perturbations to weights
        for param in self.wav2vec2.parameters():
            if param.dim() > 1:  # Only perturb weight matrices
                noise = torch.randn_like(param) * fine_tune_strength * param.std()
                param.data += noise
    
    def forward(self, input_values, attention_mask=None, **kwargs):
        return self.wav2vec2(input_values=input_values, attention_mask=attention_mask, **kwargs)

class AdaptiveWav2Vec2(nn.Module):
    """
    Adaptive Wav2Vec2 with online learning
    Uses gradient descent during inference
    """
    def __init__(self, wav2vec2_model, learning_rate=0.001):
        super().__init__()
        self.wav2vec2 = wav2vec2_model
        self.learning_rate = learning_rate
        self.optimizer = None
        
    def forward(self, input_values, attention_mask=None, **kwargs):
        # Enable gradients for online learning
        self.wav2vec2.train()
        
        # Forward pass
        outputs = self.wav2vec2(input_values=input_values, attention_mask=attention_mask, **kwargs)
        
        # Simple online learning: minimize prediction entropy
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
        
        # Backward pass and update
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.wav2vec2.parameters(), lr=self.learning_rate)
        
        self.optimizer.zero_grad()
        entropy.backward()
        self.optimizer.step()
        
        # Switch back to eval mode
        self.wav2vec2.eval()
        
        return outputs

def run_comprehensive_comparison():
    """
    Run comprehensive comparison of different ASR methods
    """
    
    # Configuration
    wav2vec2_model_id = "facebook/wav2vec2-base-960h"
    num_samples = 20  # More samples for better comparison
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Comprehensive ASR Method Comparison")
    print(f"Using device: {device}")
    print(f"Model: {wav2vec2_model_id}")
    print(f"Samples: {num_samples}")
    print("=" * 80)
    
    # Load base model and processor
    processor = Wav2Vec2Processor.from_pretrained(wav2vec2_model_id)
    base_wav2vec2 = Wav2Vec2ForCTC.from_pretrained(wav2vec2_model_id)
    base_wav2vec2.eval()
    
    # Initialize all methods
    methods = {
        'Baseline_Wav2Vec2': BaselineWav2Vec2(base_wav2vec2),
        'Chunk_TTT': Wav2Vec2WithLaCT(
            wav2vec2_model=base_wav2vec2,
            num_lact_layers=2,
            lact_hidden_size=768,
            num_lact_heads=4,
            lact_chunk_size=512,
            lact_lr_scale=0.01,
            use_momentum=True
        ),
        'Token_TTT': TokenBasedTTT(
            wav2vec2_model=base_wav2vec2,
            num_ttt_layers=2,
            hidden_size=768,
            lr_scale=0.01
        ),
        'Fine_Tuned': FineTunedWav2Vec2(
            wav2vec2_model=base_wav2vec2,
            fine_tune_strength=0.1
        ),
        'Adaptive_Online': AdaptiveWav2Vec2(
            wav2vec2_model=base_wav2vec2,
            learning_rate=0.001
        )
    }
    
    # Move to device
    for name, model in methods.items():
        model.to(device)
        model.eval()
        print(f"✓ Initialized {name}")
    
    # Load data
    print("\nLoading LibriSpeech data...")
    loader = get_librispeech_loader("./data", url="test-clean", batch_size=1)
    
    # Results storage
    results = []
    
    print(f"\nStarting evaluation...")
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
        
        # Evaluate each method
        sample_results = {'sample_id': i + 1, 'reference': reference}
        
        for method_name, model in methods.items():
            print(f"  Running {method_name}...")
            
            # Reset fast weights for TTT methods
            if hasattr(model, 'reset_fast_weights'):
                model.reset_fast_weights()
            
            # Time the inference
            start_time = time.time()
            
            try:
                with torch.no_grad():
                    outputs = model(input_values, attention_mask=attention_mask)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs['logits']
                    pred_ids = torch.argmax(logits, dim=-1)
                    hypothesis = processor.batch_decode(pred_ids)[0].lower().strip()
                
                inference_time = time.time() - start_time
                
                # Compute metrics
                wer = compute_wer(reference, hypothesis)
                cer = compute_cer(reference, hypothesis)
                
                # Store results
                sample_results.update({
                    f'{method_name}_hypothesis': hypothesis,
                    f'{method_name}_wer': wer,
                    f'{method_name}_cer': cer,
                    f'{method_name}_time': inference_time
                })
                
                print(f"    ✓ WER: {wer:.3f}, CER: {cer:.3f}, Time: {inference_time:.3f}s")
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                sample_results.update({
                    f'{method_name}_hypothesis': 'ERROR',
                    f'{method_name}_wer': 1.0,
                    f'{method_name}_cer': 1.0,
                    f'{method_name}_time': 0.0
                })
        
        # Log hardware metrics
        gpu_memory = log_hardware_metrics()
        sample_results['gpu_memory_mb'] = gpu_memory
        
        results.append(sample_results)
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        print("-" * 80)
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv("comprehensive_asr_comparison.csv", index=False)
    
    # Print summary statistics
    print_summary_statistics(df)
    
    return df

def print_summary_statistics(df):
    """Print comprehensive summary statistics"""
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE ASR METHOD COMPARISON RESULTS")
    print("=" * 80)
    
    methods = ['Baseline_Wav2Vec2', 'Chunk_TTT', 'Token_TTT', 'Fine_Tuned', 'Adaptive_Online']
    
    print(f"Number of samples: {len(df)}")
    print("\n" + "-" * 80)
    
    # WER Comparison
    print("WORD ERROR RATE (WER) COMPARISON:")
    print("-" * 40)
    for method in methods:
        wer_col = f'{method}_wer'
        if wer_col in df.columns:
            mean_wer = df[wer_col].mean()
            std_wer = df[wer_col].std()
            print(f"{method:20s}: {mean_wer:.3f} ± {std_wer:.3f}")
    
    # CER Comparison
    print("\nCHARACTER ERROR RATE (CER) COMPARISON:")
    print("-" * 40)
    for method in methods:
        cer_col = f'{method}_cer'
        if cer_col in df.columns:
            mean_cer = df[cer_col].mean()
            std_cer = df[cer_col].std()
            print(f"{method:20s}: {mean_cer:.3f} ± {std_cer:.3f}")
    
    # Time Comparison
    print("\nINFERENCE TIME COMPARISON:")
    print("-" * 40)
    baseline_time = df['Baseline_Wav2Vec2_time'].mean()
    for method in methods:
        time_col = f'{method}_time'
        if time_col in df.columns:
            mean_time = df[time_col].mean()
            overhead = (mean_time - baseline_time) / baseline_time * 100
            print(f"{method:20s}: {mean_time:.3f}s ({overhead:+.1f}% overhead)")
    
    # Performance Ranking
    print("\nPERFORMANCE RANKING (by WER):")
    print("-" * 40)
    method_wer = []
    for method in methods:
        wer_col = f'{method}_wer'
        if wer_col in df.columns:
            mean_wer = df[wer_col].mean()
            method_wer.append((method, mean_wer))
    
    method_wer.sort(key=lambda x: x[1])  # Sort by WER (lower is better)
    for i, (method, wer) in enumerate(method_wer):
        print(f"{i+1}. {method:20s}: {wer:.3f}")
    
    # Efficiency Ranking
    print("\nEFFICIENCY RANKING (by inference time):")
    print("-" * 40)
    method_time = []
    for method in methods:
        time_col = f'{method}_time'
        if time_col in df.columns:
            mean_time = df[time_col].mean()
            method_time.append((method, mean_time))
    
    method_time.sort(key=lambda x: x[1])  # Sort by time (lower is better)
    for i, (method, time_val) in enumerate(method_time):
        print(f"{i+1}. {method:20s}: {time_val:.3f}s")
    
    print(f"\nResults saved to: comprehensive_asr_comparison.csv")

def analyze_method_characteristics():
    """Analyze characteristics of different methods"""
    
    print("\n" + "=" * 80)
    print("METHOD CHARACTERISTICS ANALYSIS")
    print("=" * 80)
    
    characteristics = {
        'Baseline_Wav2Vec2': {
            'TTT_Type': 'None',
            'Update_Frequency': 'Never',
            'Memory_Overhead': 'Low',
            'Computational_Overhead': 'None',
            'Adaptation_Speed': 'None',
            'Best_For': 'Standard inference'
        },
        'Chunk_TTT': {
            'TTT_Type': 'Chunk-based',
            'Update_Frequency': 'Every 512 tokens',
            'Memory_Overhead': 'Medium',
            'Computational_Overhead': 'Medium',
            'Adaptation_Speed': 'Moderate',
            'Best_For': 'Long sequences, chunk-level adaptation'
        },
        'Token_TTT': {
            'TTT_Type': 'Token-based',
            'Update_Frequency': 'Every token',
            'Memory_Overhead': 'Low',
            'Computational_Overhead': 'High',
            'Adaptation_Speed': 'Fast',
            'Best_For': 'Fine-grained adaptation, short sequences'
        },
        'Fine_Tuned': {
            'TTT_Type': 'Pre-trained adaptation',
            'Update_Frequency': 'Pre-computed',
            'Memory_Overhead': 'None',
            'Computational_Overhead': 'None',
            'Adaptation_Speed': 'Slow (requires training)',
            'Best_For': 'Domain-specific tasks'
        },
        'Adaptive_Online': {
            'TTT_Type': 'Online learning',
            'Update_Frequency': 'Every forward pass',
            'Memory_Overhead': 'High (gradients)',
            'Computational_Overhead': 'Very High',
            'Adaptation_Speed': 'Very Fast',
            'Best_For': 'Real-time adaptation, streaming'
        }
    }
    
    # Create comparison table
    df_chars = pd.DataFrame(characteristics).T
    print(df_chars.to_string())
    
    return df_chars

if __name__ == "__main__":
    # Run comprehensive comparison
    results_df = run_comprehensive_comparison()
    
    # Analyze method characteristics
    characteristics_df = analyze_method_characteristics()
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE!")
    print("=" * 80)
    print("Files generated:")
    print("- comprehensive_asr_comparison.csv (detailed results)")
    print("- Use plot_comprehensive_results.py to visualize") 