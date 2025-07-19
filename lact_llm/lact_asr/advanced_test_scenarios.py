import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from lact_llm.lact_asr.data import get_librispeech_loader
from lact_llm.lact_asr.utils import compute_wer, compute_cer, log_hardware_metrics
from lact_llm.lact_asr.modeling_asr_wav2vec2_lact import Wav2Vec2WithLaCT, BaselineWav2Vec2
import pandas as pd
import time
import gc
import numpy as np
from typing import Dict, List, Tuple, Any

class AdvancedTestScenarios:
    """
    Advanced testing framework for comprehensive TTT evaluation
    """
    
    def __init__(self, wav2vec2_model_id="facebook/wav2vec2-base-960h"):
        self.wav2vec2_model_id = wav2vec2_model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load models
        self.processor = Wav2Vec2Processor.from_pretrained(wav2vec2_model_id)
        self.base_model = Wav2Vec2ForCTC.from_pretrained(wav2vec2_model_id)
        self.base_model.eval()
        
        # Initialize methods
        self.methods = {
            'Baseline': BaselineWav2Vec2(self.base_model),
            'Chunk_TTT': Wav2Vec2WithLaCT(
                wav2vec2_model=self.base_model,
                num_lact_layers=2,
                lact_hidden_size=768,
                num_lact_heads=4,
                lact_chunk_size=512,
                lact_lr_scale=0.01,
                use_momentum=True
            )
        }
        
        # Move to device
        for name, model in self.methods.items():
            model.to(self.device)
            model.eval()
    
    def test_scenario_1_clean_audio(self, num_samples=50):
        """
        Test Scenario 1: Clean Audio (LibriSpeech test-clean)
        Tests basic functionality on high-quality audio
        """
        print("=" * 60)
        print("TEST SCENARIO 1: Clean Audio (LibriSpeech test-clean)")
        print("=" * 60)
        
        loader = get_librispeech_loader("./data", url="test-clean", batch_size=1)
        results = self._run_evaluation(loader, num_samples, "clean_audio")
        
        print(f"Results: {len(results)} samples evaluated")
        self._print_scenario_summary(results, "Clean Audio")
        
        return results
    
    def test_scenario_2_noisy_audio(self, num_samples=30):
        """
        Test Scenario 2: Noisy Audio (LibriSpeech test-other)
        Tests robustness to background noise and poor recording conditions
        """
        print("=" * 60)
        print("TEST SCENARIO 2: Noisy Audio (LibriSpeech test-other)")
        print("=" * 60)
        
        loader = get_librispeech_loader("./data", url="test-other", batch_size=1)
        results = self._run_evaluation(loader, num_samples, "noisy_audio")
        
        print(f"Results: {len(results)} samples evaluated")
        self._print_scenario_summary(results, "Noisy Audio")
        
        return results
    
    def test_scenario_3_long_sequences(self, num_samples=20):
        """
        Test Scenario 3: Long Sequences
        Tests TTT benefits on longer audio sequences where chunk-based adaptation should help
        """
        print("=" * 60)
        print("TEST SCENARIO 3: Long Sequences")
        print("=" * 60)
        
        loader = get_librispeech_loader("./data", url="test-clean", batch_size=1)
        results = []
        
        for i, (waveform, sample_rate, transcript) in enumerate(loader):
            if i >= num_samples:
                break
            
            # Only keep longer sequences (>10 seconds)
            duration = waveform.shape[1] / sample_rate[0]
            if duration < 10.0:  # Skip short sequences
                continue
            
            print(f"Processing long sequence {len(results)+1}: {duration:.1f}s")
            
            result = self._evaluate_single_sample(waveform, sample_rate, transcript, "long_sequences")
            results.append(result)
            
            if len(results) >= num_samples:
                break
        
        print(f"Results: {len(results)} long sequences evaluated")
        self._print_scenario_summary(results, "Long Sequences")
        
        return results
    
    def test_scenario_4_domain_adaptation(self, num_samples=25):
        """
        Test Scenario 4: Domain Adaptation
        Tests how well TTT adapts to different speaking styles/domains
        """
        print("=" * 60)
        print("TEST SCENARIO 4: Domain Adaptation")
        print("=" * 60)
        
        # Test on different LibriSpeech subsets to simulate domain shift
        results = []
        
        # Test on test-clean (clean domain)
        clean_loader = get_librispeech_loader("./data", url="test-clean", batch_size=1)
        clean_results = self._run_evaluation(clean_loader, num_samples//2, "domain_clean")
        results.extend(clean_results)
        
        # Test on test-other (noisy domain)
        noisy_loader = get_librispeech_loader("./data", url="test-other", batch_size=1)
        noisy_results = self._run_evaluation(noisy_loader, num_samples//2, "domain_noisy")
        results.extend(noisy_results)
        
        print(f"Results: {len(results)} samples evaluated across domains")
        self._print_scenario_summary(results, "Domain Adaptation")
        
        return results
    
    def test_scenario_5_computational_efficiency(self, num_samples=100):
        """
        Test Scenario 5: Computational Efficiency
        Detailed analysis of computational overhead and memory usage
        """
        print("=" * 60)
        print("TEST SCENARIO 5: Computational Efficiency")
        print("=" * 60)
        
        loader = get_librispeech_loader("./data", url="test-clean", batch_size=1)
        results = []
        
        for i, (waveform, sample_rate, transcript) in enumerate(loader):
            if i >= num_samples:
                break
            
            print(f"Efficiency test {i+1}/{num_samples}")
            
            # Detailed timing and memory analysis
            result = self._evaluate_with_detailed_metrics(waveform, sample_rate, transcript)
            results.append(result)
        
        print(f"Results: {len(results)} efficiency tests completed")
        self._print_efficiency_analysis(results)
        
        return results
    
    def test_scenario_6_chunk_size_ablation(self, chunk_sizes=[128, 256, 512, 1024], num_samples=20):
        """
        Test Scenario 6: Chunk Size Ablation Study
        Tests how different chunk sizes affect performance
        """
        print("=" * 60)
        print("TEST SCENARIO 6: Chunk Size Ablation Study")
        print("=" * 60)
        
        loader = get_librispeech_loader("./data", url="test-clean", batch_size=1)
        all_results = {}
        
        for chunk_size in chunk_sizes:
            print(f"\nTesting chunk size: {chunk_size}")
            
            # Create model with specific chunk size
            chunk_model = Wav2Vec2WithLaCT(
                wav2vec2_model=self.base_model,
                num_lact_layers=2,
                lact_hidden_size=768,
                num_lact_heads=4,
                lact_chunk_size=chunk_size,
                lact_lr_scale=0.01,
                use_momentum=True
            )
            chunk_model.to(self.device)
            chunk_model.eval()
            
            # Temporarily replace Chunk_TTT method
            original_chunk_model = self.methods['Chunk_TTT']
            self.methods['Chunk_TTT'] = chunk_model
            
            # Run evaluation
            results = []
            for i, (waveform, sample_rate, transcript) in enumerate(loader):
                if i >= num_samples:
                    break
                
                result = self._evaluate_single_sample(waveform, sample_rate, transcript, f"chunk_{chunk_size}")
                results.append(result)
            
            all_results[f"chunk_{chunk_size}"] = results
            
            # Restore original model
            self.methods['Chunk_TTT'] = original_chunk_model
        
        print(f"Results: {len(chunk_sizes)} chunk sizes tested")
        self._print_chunk_size_analysis(all_results)
        
        return all_results
    
    def _run_evaluation(self, loader, num_samples, scenario_name):
        """Run evaluation on a dataset"""
        results = []
        
        for i, (waveform, sample_rate, transcript) in enumerate(loader):
            if i >= num_samples:
                break
            
            result = self._evaluate_single_sample(waveform, sample_rate, transcript, scenario_name)
            results.append(result)
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        
        return results
    
    def _evaluate_single_sample(self, waveform, sample_rate, transcript, scenario_name):
        """Evaluate a single audio sample with all methods"""
        
        # Prepare input
        input_wav = waveform[0].squeeze().numpy()
        reference = transcript[0].lower().strip()
        
        # Process with Wav2Vec2 processor
        inputs = self.processor(
            input_wav, 
            sampling_rate=sample_rate[0], 
            return_tensors="pt", 
            padding=True
        )
        input_values = inputs["input_values"].to(self.device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Evaluate each method
        sample_results = {
            'scenario': scenario_name,
            'reference': reference,
            'audio_duration': waveform.shape[1] / sample_rate[0],
            'sequence_length': input_values.shape[1]
        }
        
        for method_name, model in self.methods.items():
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
                    hypothesis = self.processor.batch_decode(pred_ids)[0].lower().strip()
                
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
                
            except Exception as e:
                print(f"Error with {method_name}: {e}")
                sample_results.update({
                    f'{method_name}_hypothesis': 'ERROR',
                    f'{method_name}_wer': 1.0,
                    f'{method_name}_cer': 1.0,
                    f'{method_name}_time': 0.0
                })
        
        # Log hardware metrics
        gpu_memory = log_hardware_metrics()
        sample_results['gpu_memory_mb'] = gpu_memory
        
        return sample_results
    
    def _evaluate_with_detailed_metrics(self, waveform, sample_rate, transcript):
        """Evaluate with detailed computational metrics"""
        
        # Similar to _evaluate_single_sample but with more detailed timing
        result = self._evaluate_single_sample(waveform, sample_rate, transcript, "efficiency_test")
        
        # Add detailed memory profiling
        if torch.cuda.is_available():
            result['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**2
            result['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024**2
            result['gpu_memory_max'] = torch.cuda.max_memory_allocated() / 1024**2
        
        return result
    
    def _print_scenario_summary(self, results, scenario_name):
        """Print summary statistics for a test scenario"""
        
        if not results:
            print("No results to analyze")
            return
        
        df = pd.DataFrame(results)
        
        print(f"\n{scenario_name} - Summary Statistics:")
        print("-" * 40)
        
        methods = ['Baseline', 'Chunk_TTT']
        
        for method in methods:
            wer_col = f'{method}_wer'
            cer_col = f'{method}_cer'
            time_col = f'{method}_time'
            
            if wer_col in df.columns:
                mean_wer = df[wer_col].mean()
                std_wer = df[wer_col].std()
                mean_cer = df[cer_col].mean()
                mean_time = df[time_col].mean()
                
                print(f"{method:12s}: WER={mean_wer:.3f}±{std_wer:.3f}, CER={mean_cer:.3f}, Time={mean_time:.3f}s")
        
        # Calculate improvement
        if 'Baseline_wer' in df.columns and 'Chunk_TTT_wer' in df.columns:
            baseline_wer = df['Baseline_wer'].mean()
            chunk_wer = df['Chunk_TTT_wer'].mean()
            improvement = (baseline_wer - chunk_wer) / baseline_wer * 100
            
            print(f"\nChunk TTT vs Baseline improvement: {improvement:+.1f}%")
    
    def _print_efficiency_analysis(self, results):
        """Print detailed efficiency analysis"""
        
        df = pd.DataFrame(results)
        
        print("\nComputational Efficiency Analysis:")
        print("-" * 40)
        
        baseline_time = df['Baseline_time'].mean()
        chunk_time = df['Chunk_TTT_time'].mean()
        time_overhead = (chunk_time - baseline_time) / baseline_time * 100
        
        print(f"Baseline inference time: {baseline_time:.3f}s")
        print(f"Chunk TTT inference time: {chunk_time:.3f}s")
        print(f"Time overhead: {time_overhead:+.1f}%")
        
        if 'gpu_memory_mb' in df.columns:
            mean_memory = df['gpu_memory_mb'].mean()
            print(f"Average GPU memory usage: {mean_memory:.1f} MB")
    
    def _print_chunk_size_analysis(self, all_results):
        """Print chunk size ablation analysis"""
        
        print("\nChunk Size Ablation Study:")
        print("-" * 40)
        
        for chunk_size, results in all_results.items():
            df = pd.DataFrame(results)
            
            if 'Chunk_TTT_wer' in df.columns:
                mean_wer = df['Chunk_TTT_wer'].mean()
                mean_time = df['Chunk_TTT_time'].mean()
                print(f"{chunk_size:12s}: WER={mean_wer:.3f}, Time={mean_time:.3f}s")

def run_all_test_scenarios():
    """Run all test scenarios"""
    
    print("Advanced Test Scenarios for Chunk-based TTT Evaluation")
    print("=" * 80)
    
    # Initialize test framework
    tester = AdvancedTestScenarios()
    
    # Run all scenarios
    results = {}
    
    print("\n1. Testing Clean Audio...")
    results['clean_audio'] = tester.test_scenario_1_clean_audio(num_samples=50)
    
    print("\n2. Testing Noisy Audio...")
    results['noisy_audio'] = tester.test_scenario_2_noisy_audio(num_samples=30)
    
    print("\n3. Testing Long Sequences...")
    results['long_sequences'] = tester.test_scenario_3_long_sequences(num_samples=20)
    
    print("\n4. Testing Domain Adaptation...")
    results['domain_adaptation'] = tester.test_scenario_4_domain_adaptation(num_samples=25)
    
    print("\n5. Testing Computational Efficiency...")
    results['efficiency'] = tester.test_scenario_5_computational_efficiency(num_samples=50)
    
    print("\n6. Testing Chunk Size Ablation...")
    results['chunk_ablation'] = tester.test_scenario_6_chunk_size_ablation(num_samples=15)
    
    # Save all results
    all_results = []
    for scenario_name, scenario_results in results.items():
        if isinstance(scenario_results, list):
            all_results.extend(scenario_results)
        else:
            # Handle chunk ablation results (dict of lists)
            for chunk_size, chunk_results in scenario_results.items():
                all_results.extend(chunk_results)
    
    df = pd.DataFrame(all_results)
    df.to_csv("advanced_test_scenarios_results.csv", index=False)
    
    print(f"\nAll test scenarios completed!")
    print(f"Results saved to: advanced_test_scenarios_results.csv")
    print(f"Total samples evaluated: {len(all_results)}")
    
    return results

if __name__ == "__main__":
    run_all_test_scenarios() 