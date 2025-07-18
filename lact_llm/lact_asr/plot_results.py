import pandas as pd
import matplotlib.pyplot as plt

# Load results
results = pd.read_csv("results_asr_experiment.csv")

methods = ["BASE", "CHUNK_TTT", "TOKEN_TTT"]

# Plot WER
plt.figure(figsize=(10, 5))
for method in methods:
    plt.plot(results["Sample"], results[f"WER_{method}"], label=f"WER {method}")
plt.xlabel("Sample")
plt.ylabel("WER")
plt.title("WER per Sample")
plt.legend()
plt.tight_layout()
plt.savefig("wer_per_sample.png")
plt.show()

# Plot CER
plt.figure(figsize=(10, 5))
for method in methods:
    plt.plot(results["Sample"], results[f"CER_{method}"], label=f"CER {method}")
plt.xlabel("Sample")
plt.ylabel("CER")
plt.title("CER per Sample")
plt.legend()
plt.tight_layout()
plt.savefig("cer_per_sample.png")
plt.show()

# Print average WER/CER
for method in methods:
    print(f"{method}: Avg WER = {results[f'WER_{method}'].mean():.3f}, Avg CER = {results[f'CER_{method}'].mean():.3f}") 