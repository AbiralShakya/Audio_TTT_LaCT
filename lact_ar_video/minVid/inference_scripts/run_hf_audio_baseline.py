import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import soundfile as sf

# Use MusicGen (Hugging Face official support)
model_id = "facebook/musicgen-small"
processor = AutoProcessor.from_pretrained(model_id)
model = MusicgenForConditionalGeneration.from_pretrained(model_id)

prompt = "A dog barking in a park with children playing."
inputs = processor(text=prompt, return_tensors="pt")

with torch.no_grad():
    audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)

# audio_values: (batch, channels, samples)
sampling_rate = model.config.audio_encoder.sampling_rate
sf.write("musicgen_baseline_output.wav", audio_values[0, 0].cpu().numpy(), sampling_rate)
print(f"Audio generated and saved as musicgen_baseline_output.wav at {sampling_rate} Hz") 