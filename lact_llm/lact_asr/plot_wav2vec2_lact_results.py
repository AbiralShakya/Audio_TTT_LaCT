import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_wav2vec2_lact_comparison(csv_file="results_wav2vec2_lact_comparison.csv"):
    """
    Plot comparison results between baseline Wav2Vec2 and Wav2Vec2 with LaCT
    """
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Results file {csv_file} not found. Please run the experiment first.")
        return
    
    # Set style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Wav2Vec2 vs Wav2Vec2+LaCT Comparison', fontsize=16, fontweight='bold')
    
    # 1. WER Comparison
    ax1 = axes[0, 0]
    x = np.arange(len(df))
    width = 0.35
    
    ax1.bar(x - width/2, df['baseline_wer'], width, label='Baseline Wav2Vec2', alpha=0.8, color='skyblue')
    ax1.bar(x + width/2, df['lact_wer'], width, label='Wav2Vec2 + LaCT', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('Sample ID')
    ax1.set_ylabel('Word Error Rate (WER)')
    ax1.set_title('WER Comparison per Sample')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. CER Comparison
    ax2 = axes[0, 1]
    ax2.bar(x - width/2, df['baseline_cer'], width, label='Baseline Wav2Vec2', alpha=0.8, color='skyblue')
    ax2.bar(x + width/2, df['lact_cer'], width, label='Wav2Vec2 + LaCT', alpha=0.8, color='lightcoral')
    
    ax2.set_xlabel('Sample ID')
    ax2.set_ylabel('Character Error Rate (CER)')
    ax2.set_title('CER Comparison per Sample')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Inference Time Comparison
    ax3 = axes[1, 0]
    ax3.bar(x - width/2, df['baseline_time'], width, label='Baseline Wav2Vec2', alpha=0.8, color='skyblue')
    ax3.bar(x + width/2, df['lact_time'], width, label='Wav2Vec2 + LaCT', alpha=0.8, color='lightcoral')
    
    ax3.set_xlabel('Sample ID')
    ax3.set_ylabel('Inference Time (seconds)')
    ax3.set_title('Inference Time Comparison per Sample')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary Statistics
    ax4 = axes[1, 1]
    
    # Calculate averages
    metrics = ['WER', 'CER', 'Time (s)']
    baseline_avg = [df['baseline_wer'].mean(), df['baseline_cer'].mean(), df['baseline_time'].mean()]
    lact_avg = [df['lact_wer'].mean(), df['lact_cer'].mean(), df['lact_time'].mean()]
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    ax4.bar(x_pos - width/2, baseline_avg, width, label='Baseline Wav2Vec2', alpha=0.8, color='skyblue')
    ax4.bar(x_pos + width/2, lact_avg, width, label='Wav2Vec2 + LaCT', alpha=0.8, color='lightcoral')
    
    ax4.set_xlabel('Metrics')
    ax4.set_ylabel('Average Values')
    ax4.set_title('Average Performance Comparison')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (baseline_val, lact_val) in enumerate(zip(baseline_avg, lact_avg)):
        ax4.text(i - width/2, baseline_val + 0.001, f'{baseline_val:.3f}', ha='center', va='bottom', fontsize=8)
        ax4.text(i + width/2, lact_val + 0.001, f'{lact_val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('wav2vec2_lact_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Number of samples: {len(df)}")
    print(f"\nAverage WER:")
    print(f"  Baseline Wav2Vec2: {df['baseline_wer'].mean():.3f} ± {df['baseline_wer'].std():.3f}")
    print(f"  Wav2Vec2 + LaCT:   {df['lact_wer'].mean():.3f} ± {df['lact_wer'].std():.3f}")
    
    print(f"\nAverage CER:")
    print(f"  Baseline Wav2Vec2: {df['baseline_cer'].mean():.3f} ± {df['baseline_cer'].std():.3f}")
    print(f"  Wav2Vec2 + LaCT:   {df['lact_cer'].mean():.3f} ± {df['lact_cer'].std():.3f}")
    
    print(f"\nAverage Inference Time:")
    print(f"  Baseline Wav2Vec2: {df['baseline_time'].mean():.3f}s ± {df['baseline_time'].std():.3f}s")
    print(f"  Wav2Vec2 + LaCT:   {df['lact_time'].mean():.3f}s ± {df['lact_time'].std():.3f}s")
    
    # Calculate improvements
    wer_improvement = df['baseline_wer'].mean() - df['lact_wer'].mean()
    cer_improvement = df['baseline_cer'].mean() - df['lact_cer'].mean()
    time_overhead = df['lact_time'].mean() - df['baseline_time'].mean()
    
    print(f"\nLaCT vs Baseline Improvements:")
    print(f"  WER Improvement: {wer_improvement:+.3f} ({wer_improvement/df['baseline_wer'].mean()*100:+.1f}%)")
    print(f"  CER Improvement: {cer_improvement:+.3f} ({cer_improvement/df['baseline_cer'].mean()*100:+.1f}%)")
    print(f"  Time Overhead: {time_overhead:+.3f}s ({time_overhead/df['baseline_time'].mean()*100:+.1f}%)")
    
    print(f"\nPlot saved as: wav2vec2_lact_comparison.png")

if __name__ == "__main__":
    plot_wav2vec2_lact_comparison() 