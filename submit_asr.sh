# Run comprehensive comparison
cd lact_llm/lact_asr
python comprehensive_asr_comparison.py

# Visualize results
python plot_comprehensive_results.py

# Analyze specific metrics
python -c "
import pandas as pd
df = pd.read_csv('comprehensive_asr_comparison.csv')
print('Chunk TTT improvement:', 
      (df['Baseline_Wav2Vec2_wer'].mean() - df['Chunk_TTT_wer'].mean()) / df['Baseline_Wav2Vec2_wer'].mean() * 100, '%')
"