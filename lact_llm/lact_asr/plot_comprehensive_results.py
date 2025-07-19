import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict

def plot_comprehensive_asr_comparison(csv_file="comprehensive_asr_comparison.csv"):
    """
    Create comprehensive visualization of ASR method comparison
    """
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Results file {csv_file} not found. Please run comprehensive_asr_comparison.py first.")
        return
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Define methods and colors
    methods = ['Baseline_Wav2Vec2', 'Chunk_TTT', 'Token_TTT', 'Fine_Tuned', 'Adaptive_Online']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8B5A3C']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. WER Comparison (Bar plot)
    ax1 = plt.subplot(3, 3, 1)
    wer_data = []
    for method in methods:
        wer_col = f'{method}_wer'
        if wer_col in df.columns:
            mean_wer = df[wer_col].mean()
            std_wer = df[wer_col].std()
            wer_data.append((method.replace('_', ' '), mean_wer, std_wer))
    
    methods_clean, means, stds = zip(*wer_data)
    bars = ax1.bar(methods_clean, means, yerr=stds, capsize=5, color=colors[:len(methods_clean)], alpha=0.8)
    ax1.set_title('Word Error Rate (WER) Comparison', fontweight='bold', fontsize=12)
    ax1.set_ylabel('WER (lower is better)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, mean_val in zip(bars, means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 2. CER Comparison (Bar plot)
    ax2 = plt.subplot(3, 3, 2)
    cer_data = []
    for method in methods:
        cer_col = f'{method}_cer'
        if cer_col in df.columns:
            mean_cer = df[cer_col].mean()
            std_cer = df[cer_col].std()
            cer_data.append((method.replace('_', ' '), mean_cer, std_cer))
    
    methods_clean, means, stds = zip(*cer_data)
    bars = ax2.bar(methods_clean, means, yerr=stds, capsize=5, color=colors[:len(methods_clean)], alpha=0.8)
    ax2.set_title('Character Error Rate (CER) Comparison', fontweight='bold', fontsize=12)
    ax2.set_ylabel('CER (lower is better)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, mean_val in zip(bars, means):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 3. Inference Time Comparison
    ax3 = plt.subplot(3, 3, 3)
    time_data = []
    baseline_time = df['Baseline_Wav2Vec2_time'].mean()
    for method in methods:
        time_col = f'{method}_time'
        if time_col in df.columns:
            mean_time = df[time_col].mean()
            overhead = (mean_time - baseline_time) / baseline_time * 100
            time_data.append((method.replace('_', ' '), mean_time, overhead))
    
    methods_clean, times, overheads = zip(*time_data)
    bars = ax3.bar(methods_clean, times, color=colors[:len(methods_clean)], alpha=0.8)
    ax3.set_title('Inference Time Comparison', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Time (seconds)')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add overhead labels
    for bar, overhead in zip(bars, overheads):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{overhead:+.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 4. WER vs Time Scatter Plot
    ax4 = plt.subplot(3, 3, 4)
    for i, method in enumerate(methods):
        wer_col = f'{method}_wer'
        time_col = f'{method}_time'
        if wer_col in df.columns and time_col in df.columns:
            ax4.scatter(df[time_col], df[wer_col], 
                       label=method.replace('_', ' '), 
                       color=colors[i], alpha=0.7, s=50)
    
    ax4.set_xlabel('Inference Time (seconds)')
    ax4.set_ylabel('Word Error Rate (WER)')
    ax4.set_title('WER vs Inference Time', fontweight='bold', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Performance per Sample (Line plot)
    ax5 = plt.subplot(3, 3, 5)
    for i, method in enumerate(methods):
        wer_col = f'{method}_wer'
        if wer_col in df.columns:
            ax5.plot(df['sample_id'], df[wer_col], 
                    label=method.replace('_', ' '), 
                    color=colors[i], alpha=0.8, linewidth=2, marker='o', markersize=4)
    
    ax5.set_xlabel('Sample ID')
    ax5.set_ylabel('Word Error Rate (WER)')
    ax5.set_title('WER per Sample', fontweight='bold', fontsize=12)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Method Ranking Heatmap
    ax6 = plt.subplot(3, 3, 6)
    
    # Calculate rankings for each metric
    metrics = ['wer', 'cer', 'time']
    ranking_data = []
    
    for metric in metrics:
        metric_rankings = []
        for method in methods:
            col = f'{method}_{metric}'
            if col in df.columns:
                mean_val = df[col].mean()
                metric_rankings.append((method.replace('_', ' '), mean_val))
        
        # Sort by value (lower is better for WER/CER, lower is better for time)
        metric_rankings.sort(key=lambda x: x[1])
        ranking_data.append([rank[0] for rank in metric_rankings])
    
    # Create heatmap
    ranking_df = pd.DataFrame(ranking_data, 
                             index=['WER Rank', 'CER Rank', 'Time Rank'],
                             columns=[f'Rank {i+1}' for i in range(len(methods))])
    
    sns.heatmap(ranking_df, annot=True, fmt='', cmap='YlOrRd', ax=ax6, cbar=False)
    ax6.set_title('Method Rankings', fontweight='bold', fontsize=12)
    
    # 7. Performance Distribution (Box plot)
    ax7 = plt.subplot(3, 3, 7)
    wer_data_for_box = []
    labels = []
    for method in methods:
        wer_col = f'{method}_wer'
        if wer_col in df.columns:
            wer_data_for_box.append(df[wer_col].values)
            labels.append(method.replace('_', ' '))
    
    box_plot = ax7.boxplot(wer_data_for_box, labels=labels, patch_artist=True)
    for patch, color in zip(box_plot['boxes'], colors[:len(labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax7.set_title('WER Distribution', fontweight='bold', fontsize=12)
    ax7.set_ylabel('Word Error Rate (WER)')
    ax7.tick_params(axis='x', rotation=45)
    
    # 8. Efficiency vs Accuracy Trade-off
    ax8 = plt.subplot(3, 3, 8)
    
    # Calculate efficiency (1/time) vs accuracy (1-WER)
    for i, method in enumerate(methods):
        wer_col = f'{method}_wer'
        time_col = f'{method}_time'
        if wer_col in df.columns and time_col in df.columns:
            efficiency = 1 / df[time_col].mean()  # Higher is better
            accuracy = 1 - df[wer_col].mean()     # Higher is better
            ax8.scatter(efficiency, accuracy, 
                       label=method.replace('_', ' '), 
                       color=colors[i], s=100, alpha=0.8)
    
    ax8.set_xlabel('Efficiency (1/Time)')
    ax8.set_ylabel('Accuracy (1-WER)')
    ax8.set_title('Efficiency vs Accuracy Trade-off', fontweight='bold', fontsize=12)
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Summary Statistics Table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('tight')
    ax9.axis('off')
    
    # Create summary table
    summary_data = []
    for method in methods:
        row = [method.replace('_', ' ')]
        for metric in ['wer', 'cer', 'time']:
            col = f'{method}_{metric}'
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                row.append(f'{mean_val:.3f}±{std_val:.3f}')
            else:
                row.append('N/A')
        summary_data.append(row)
    
    table = ax9.table(cellText=summary_data,
                     colLabels=['Method', 'WER', 'CER', 'Time(s)'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.3, 0.2, 0.2, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(summary_data) + 1):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#4A90E2')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#F0F0F0' if i % 2 == 0 else 'white')
    
    ax9.set_title('Summary Statistics', fontweight='bold', fontsize=12, pad=20)
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig('comprehensive_asr_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed analysis
    print_detailed_analysis(df)

def print_detailed_analysis(df):
    """Print detailed analysis of the results"""
    
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)
    
    methods = ['Baseline_Wav2Vec2', 'Chunk_TTT', 'Token_TTT', 'Fine_Tuned', 'Adaptive_Online']
    
    # 1. Best performing method for each metric
    print("BEST PERFORMING METHODS:")
    print("-" * 40)
    
    for metric, metric_name in [('wer', 'WER'), ('cer', 'CER'), ('time', 'Inference Time')]:
        best_method = None
        best_value = float('inf')
        
        for method in methods:
            col = f'{method}_{metric}'
            if col in df.columns:
                mean_val = df[col].mean()
                if mean_val < best_value:
                    best_value = mean_val
                    best_method = method
        
        if best_method:
            print(f"{metric_name:15s}: {best_method.replace('_', ' ')} ({best_value:.3f})")
    
    # 2. Statistical significance analysis
    print("\nSTATISTICAL SIGNIFICANCE (WER):")
    print("-" * 40)
    
    baseline_wer = df['Baseline_Wav2Vec2_wer'].values
    for method in methods[1:]:  # Skip baseline
        wer_col = f'{method}_wer'
        if wer_col in df.columns:
            method_wer = df[wer_col].values
            improvement = baseline_wer.mean() - method_wer.mean()
            improvement_pct = (improvement / baseline_wer.mean()) * 100
            
            print(f"{method.replace('_', ' '):20s}: {improvement:+.3f} ({improvement_pct:+.1f}%)")
    
    # 3. Trade-off analysis
    print("\nTRADE-OFF ANALYSIS:")
    print("-" * 40)
    
    # Find Pareto optimal methods
    pareto_methods = []
    for method in methods:
        wer_col = f'{method}_wer'
        time_col = f'{method}_time'
        if wer_col in df.columns and time_col in df.columns:
            wer = df[wer_col].mean()
            time = df[time_col].mean()
            pareto_methods.append((method, wer, time))
    
    # Check for Pareto dominance
    pareto_optimal = []
    for i, (method1, wer1, time1) in enumerate(pareto_methods):
        dominated = False
        for j, (method2, wer2, time2) in enumerate(pareto_methods):
            if i != j and wer2 <= wer1 and time2 <= time1 and (wer2 < wer1 or time2 < time1):
                dominated = True
                break
        if not dominated:
            pareto_optimal.append(method1)
    
    print("Pareto Optimal Methods:")
    for method in pareto_optimal:
        print(f"  - {method.replace('_', ' ')}")
    
    # 4. Recommendations
    print("\nRECOMMENDATIONS:")
    print("-" * 40)
    
    print("Based on the results:")
    print("1. For maximum accuracy: Choose method with lowest WER")
    print("2. For real-time applications: Choose method with lowest inference time")
    print("3. For balanced performance: Choose Pareto optimal method")
    print("4. For research: Focus on Chunk_TTT vs Token_TTT comparison")
    
    print(f"\nPlot saved as: comprehensive_asr_comparison.png")

def create_method_comparison_table():
    """Create a comparison table of method characteristics"""
    
    characteristics = {
        'Method': ['Baseline Wav2Vec2', 'Chunk TTT', 'Token TTT', 'Fine-tuned', 'Adaptive Online'],
        'TTT Type': ['None', 'Chunk-based', 'Token-based', 'Pre-trained', 'Online Learning'],
        'Update Frequency': ['Never', 'Every 512 tokens', 'Every token', 'Pre-computed', 'Every forward pass'],
        'Memory Overhead': ['Low', 'Medium', 'Low', 'None', 'High'],
        'Computational Overhead': ['None', 'Medium', 'High', 'None', 'Very High'],
        'Adaptation Speed': ['None', 'Moderate', 'Fast', 'Slow', 'Very Fast'],
        'Best Use Case': ['Standard inference', 'Long sequences', 'Fine-grained adaptation', 'Domain-specific', 'Real-time streaming']
    }
    
    df_chars = pd.DataFrame(characteristics)
    print("\n" + "=" * 80)
    print("METHOD CHARACTERISTICS COMPARISON")
    print("=" * 80)
    print(df_chars.to_string(index=False))
    
    return df_chars

if __name__ == "__main__":
    # Plot comprehensive results
    plot_comprehensive_asr_comparison()
    
    # Create method comparison table
    create_method_comparison_table() 