#!/bin/bash

# Quick ASR Test Script
# Fast evaluation of chunk-based TTT vs baseline

set -e  # Exit on any error

echo "=========================================="
echo "Quick ASR Test: Chunk TTT vs Baseline"
echo "=========================================="

# Configuration
EXPERIMENT_DIR="lact_llm/lact_asr"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "Timestamp: $TIMESTAMP"
echo ""

# Check if we're in the right directory
if [ ! -d "$EXPERIMENT_DIR" ]; then
    echo "❌ Error: $EXPERIMENT_DIR directory not found"
    echo "Please run this script from the project root directory"
    exit 1
fi

cd $EXPERIMENT_DIR

# Check required files
echo "📁 Checking required files..."
if [ ! -f "comprehensive_asr_comparison.py" ]; then
    echo "❌ Error: comprehensive_asr_comparison.py not found"
    exit 1
fi

if [ ! -f "plot_comprehensive_results.py" ]; then
    echo "❌ Error: plot_comprehensive_results.py not found"
    exit 1
fi

echo "✅ All files found"
echo ""

# Run basic comparison
echo "🚀 Running Basic Comparison..."
echo "This will test chunk TTT vs baseline on 20 LibriSpeech samples"
echo "----------------------------------------"

if python comprehensive_asr_comparison.py; then
    echo "✅ Basic comparison completed successfully"
else
    echo "❌ Basic comparison failed"
    exit 1
fi

echo ""

# Generate visualization
echo "📊 Generating Visualization..."
echo "----------------------------------------"

if python plot_comprehensive_results.py; then
    echo "✅ Visualization generated successfully"
else
    echo "❌ Visualization failed"
    exit 1
fi

echo ""

# Quick analysis
echo "📋 Quick Analysis..."
echo "----------------------------------------"

python -c "
import pandas as pd
try:
    df = pd.read_csv('comprehensive_asr_comparison.csv')
    print(f'✅ Loaded results: {len(df)} samples')
    
    if 'Baseline_Wav2Vec2_wer' in df.columns and 'Chunk_TTT_wer' in df.columns:
        baseline_wer = df['Baseline_Wav2Vec2_wer'].mean()
        chunk_wer = df['Chunk_TTT_wer'].mean()
        improvement = (baseline_wer - chunk_wer) / baseline_wer * 100
        
        baseline_time = df['Baseline_Wav2Vec2_time'].mean()
        chunk_time = df['Chunk_TTT_time'].mean()
        time_overhead = (chunk_time - baseline_time) / baseline_time * 100
        
        print(f'\\n📊 RESULTS:')
        print(f'Baseline WER: {baseline_wer:.3f}')
        print(f'Chunk TTT WER: {chunk_wer:.3f}')
        print(f'WER Improvement: {improvement:+.1f}%')
        print(f'Time Overhead: {time_overhead:+.1f}%')
        
        if improvement > 0:
            print(f'\\n🎉 Chunk TTT improves WER by {improvement:.1f}%!')
        else:
            print(f'\\n⚠️  Chunk TTT does not improve WER in this test')
            
    else:
        print('❌ Required columns not found in results')
        
except Exception as e:
    print(f'❌ Error reading results: {e}')
"

echo ""
echo "🎉 Quick test completed!"
echo "========================="
echo ""
echo "📊 Generated files:"
ls -la *.csv *.png 2>/dev/null || echo "No CSV/PNG files found"
echo ""
echo "📖 Next steps:"
echo "1. Review comprehensive_asr_comparison.csv for detailed results"
echo "2. View comprehensive_asr_comparison.png for visualizations"
echo "3. Run ./run_asr_experiments.sh for full evaluation"
echo "" 