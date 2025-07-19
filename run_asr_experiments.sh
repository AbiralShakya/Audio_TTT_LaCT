#!/bin/bash

# ASR Experiments Runner Script
# Comprehensive evaluation of chunk-based TTT vs other methods

set -e  # Exit on any error

echo "=========================================="
echo "ASR Experiments: Chunk TTT vs Other Methods"
echo "=========================================="

# Configuration
EXPERIMENT_DIR="lact_llm/lact_asr"
RESULTS_DIR="results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create results directory
mkdir -p $RESULTS_DIR

echo "Timestamp: $TIMESTAMP"
echo "Results will be saved to: $RESULTS_DIR/"
echo ""

# Function to check if command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "❌ Error: $1 is not installed"
        exit 1
    fi
}

# Function to check if file exists
check_file() {
    if [ ! -f "$1" ]; then
        echo "❌ Error: Required file $1 not found"
        exit 1
    fi
}

# Check prerequisites
echo "🔍 Checking prerequisites..."
check_command python
check_command pip

# Check if we're in the right directory
if [ ! -d "$EXPERIMENT_DIR" ]; then
    echo "❌ Error: $EXPERIMENT_DIR directory not found"
    echo "Please run this script from the project root directory"
    exit 1
fi

cd $EXPERIMENT_DIR

# Check required files
echo "📁 Checking required files..."
check_file "comprehensive_asr_comparison.py"
check_file "advanced_test_scenarios.py"
check_file "plot_comprehensive_results.py"
check_file "modeling_asr_wav2vec2_lact.py"

echo "✅ All prerequisites satisfied"
echo ""

# Function to run experiment with error handling
run_experiment() {
    local name=$1
    local command=$2
    local output_file=$3
    
    echo "🚀 Running: $name"
    echo "Command: $command"
    echo "Output: $output_file"
    echo "----------------------------------------"
    
    # Run the experiment and capture output
    if eval "$command" > "$output_file" 2>&1; then
        echo "✅ $name completed successfully"
        echo "Results saved to: $output_file"
    else
        echo "❌ $name failed"
        echo "Check logs in: $output_file"
        return 1
    fi
    echo ""
}

# Function to check if experiment results exist
check_results() {
    local file=$1
    if [ -f "$file" ]; then
        echo "✅ Found existing results: $file"
        return 0
    else
        echo "❌ Missing results: $file"
        return 1
    fi
}

# Start experiments
echo "🎯 Starting ASR Experiments"
echo "=========================="

# 1. Basic Comprehensive Comparison
echo "📊 Experiment 1: Basic Comprehensive Comparison"
BASIC_OUTPUT="$RESULTS_DIR/basic_comparison_${TIMESTAMP}.log"
run_experiment "Basic Comparison" \
    "python comprehensive_asr_comparison.py" \
    "$BASIC_OUTPUT"

# Check if basic comparison results were generated
if check_results "comprehensive_asr_comparison.csv"; then
    echo "📈 Basic comparison results generated successfully"
else
    echo "⚠️  Basic comparison may have failed - check logs"
fi

# 2. Advanced Test Scenarios
echo "🔬 Experiment 2: Advanced Test Scenarios"
ADVANCED_OUTPUT="$RESULTS_DIR/advanced_scenarios_${TIMESTAMP}.log"
run_experiment "Advanced Scenarios" \
    "python advanced_test_scenarios.py" \
    "$ADVANCED_OUTPUT"

# Check if advanced results were generated
if check_results "advanced_test_scenarios_results.csv"; then
    echo "📈 Advanced scenarios results generated successfully"
else
    echo "⚠️  Advanced scenarios may have failed - check logs"
fi

# 3. Generate Visualizations
echo "📊 Experiment 3: Generate Visualizations"
VISUALIZATION_OUTPUT="$RESULTS_DIR/visualization_${TIMESTAMP}.log"
run_experiment "Visualization" \
    "python plot_comprehensive_results.py" \
    "$VISUALIZATION_OUTPUT"

# Check if visualization was generated
if [ -f "comprehensive_asr_comparison.png" ]; then
    echo "📈 Visualization generated successfully"
    mv comprehensive_asr_comparison.png "$RESULTS_DIR/comprehensive_asr_comparison_${TIMESTAMP}.png"
else
    echo "⚠️  Visualization may have failed - check logs"
fi

# 4. Quick Analysis
echo "📋 Experiment 4: Quick Analysis"
ANALYSIS_OUTPUT="$RESULTS_DIR/quick_analysis_${TIMESTAMP}.log"

cat > temp_analysis.py << 'EOF'
import pandas as pd
import numpy as np

print("=" * 60)
print("QUICK ANALYSIS OF ASR EXPERIMENT RESULTS")
print("=" * 60)

# Try to load basic comparison results
try:
    df_basic = pd.read_csv('comprehensive_asr_comparison.csv')
    print(f"✅ Loaded basic comparison results: {len(df_basic)} samples")
    
    # Basic statistics
    methods = ['Baseline_Wav2Vec2', 'Chunk_TTT', 'Token_TTT', 'Fine_Tuned', 'Adaptive_Online']
    
    print("\n📊 BASIC COMPARISON RESULTS:")
    print("-" * 40)
    for method in methods:
        wer_col = f'{method}_wer'
        time_col = f'{method}_time'
        if wer_col in df_basic.columns:
            mean_wer = df_basic[wer_col].mean()
            std_wer = df_basic[wer_col].std()
            mean_time = df_basic[time_col].mean()
            print(f"{method:20s}: WER={mean_wer:.3f}±{std_wer:.3f}, Time={mean_time:.3f}s")
    
    # Calculate improvements
    if 'Baseline_Wav2Vec2_wer' in df_basic.columns and 'Chunk_TTT_wer' in df_basic.columns:
        baseline_wer = df_basic['Baseline_Wav2Vec2_wer'].mean()
        chunk_wer = df_basic['Chunk_TTT_wer'].mean()
        improvement = (baseline_wer - chunk_wer) / baseline_wer * 100
        print(f"\n🎯 Chunk TTT vs Baseline improvement: {improvement:+.1f}%")
        
        baseline_time = df_basic['Baseline_Wav2Vec2_time'].mean()
        chunk_time = df_basic['Chunk_TTT_time'].mean()
        time_overhead = (chunk_time - baseline_time) / baseline_time * 100
        print(f"⏱️  Chunk TTT time overhead: {time_overhead:+.1f}%")
    
except FileNotFoundError:
    print("❌ Basic comparison results not found")

# Try to load advanced scenarios results
try:
    df_advanced = pd.read_csv('advanced_test_scenarios_results.csv')
    print(f"\n✅ Loaded advanced scenarios results: {len(df_advanced)} samples")
    
    # Analyze by scenario
    scenarios = df_advanced['scenario'].unique()
    print(f"\n📊 ADVANCED SCENARIOS RESULTS:")
    print("-" * 40)
    
    for scenario in scenarios:
        scenario_data = df_advanced[df_advanced['scenario'] == scenario]
        if len(scenario_data) > 0:
            baseline_wer = scenario_data['Baseline_wer'].mean()
            chunk_wer = scenario_data['Chunk_TTT_wer'].mean()
            improvement = (baseline_wer - chunk_wer) / baseline_wer * 100
            print(f"{scenario:20s}: {len(scenario_data)} samples, improvement: {improvement:+.1f}%")
    
except FileNotFoundError:
    print("❌ Advanced scenarios results not found")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
EOF

run_experiment "Quick Analysis" \
    "python temp_analysis.py" \
    "$ANALYSIS_OUTPUT"

# Clean up temporary file
rm -f temp_analysis.py

# 5. Copy all results to results directory
echo "📁 Copying results to results directory..."
cp -f *.csv "$RESULTS_DIR/" 2>/dev/null || true
cp -f *.png "$RESULTS_DIR/" 2>/dev/null || true

# 6. Generate summary report
echo "📋 Generating Summary Report"
SUMMARY_REPORT="$RESULTS_DIR/experiment_summary_${TIMESTAMP}.txt"

cat > "$SUMMARY_REPORT" << EOF
ASR Experiments Summary Report
=============================
Timestamp: $TIMESTAMP
Experiment Directory: $EXPERIMENT_DIR
Results Directory: $RESULTS_DIR

Experiments Run:
1. Basic Comprehensive Comparison
   - Output: $BASIC_OUTPUT
   - Results: comprehensive_asr_comparison.csv

2. Advanced Test Scenarios
   - Output: $ADVANCED_OUTPUT
   - Results: advanced_test_scenarios_results.csv

3. Visualization Generation
   - Output: $VISUALIZATION_OUTPUT
   - Results: comprehensive_asr_comparison.png

4. Quick Analysis
   - Output: $ANALYSIS_OUTPUT

Files Generated:
$(ls -la $RESULTS_DIR/*${TIMESTAMP}* 2>/dev/null || echo "No timestamped files found")

Available Results:
$(ls -la $RESULTS_DIR/*.csv $RESULTS_DIR/*.png 2>/dev/null || echo "No CSV/PNG files found")

Next Steps:
1. Review the log files for any errors
2. Check the CSV files for detailed results
3. View the PNG files for visualizations
4. Run additional analysis as needed

EOF

echo "📋 Summary report generated: $SUMMARY_REPORT"

# 7. Final status
echo ""
echo "🎉 EXPERIMENTS COMPLETED!"
echo "========================="
echo ""
echo "📁 Results saved to: $RESULTS_DIR/"
echo "📋 Summary report: $SUMMARY_REPORT"
echo ""
echo "📊 Generated files:"
ls -la $RESULTS_DIR/*${TIMESTAMP}* 2>/dev/null || echo "No timestamped files found"
echo ""
echo "📈 Available results:"
ls -la $RESULTS_DIR/*.csv $RESULTS_DIR/*.png 2>/dev/null || echo "No CSV/PNG files found"
echo ""

# Optional: Show quick results if available
if [ -f "$RESULTS_DIR/comprehensive_asr_comparison.csv" ]; then
    echo "🔍 Quick Results Preview:"
    echo "------------------------"
    python -c "
import pandas as pd
try:
    df = pd.read_csv('$RESULTS_DIR/comprehensive_asr_comparison.csv')
    if 'Baseline_Wav2Vec2_wer' in df.columns and 'Chunk_TTT_wer' in df.columns:
        baseline_wer = df['Baseline_Wav2Vec2_wer'].mean()
        chunk_wer = df['Chunk_TTT_wer'].mean()
        improvement = (baseline_wer - chunk_wer) / baseline_wer * 100
        print(f'Chunk TTT vs Baseline WER improvement: {improvement:+.1f}%')
    else:
        print('Required columns not found in results')
except Exception as e:
    print(f'Error reading results: {e}')
"
fi

echo ""
echo "✅ All experiments completed successfully!"
echo "📖 Check the log files for detailed output"
echo "🎯 Review the CSV files for detailed results"
echo "📊 View the PNG files for visualizations" 