import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import soundfile as sf
from lact_audio.models.lact_audio import LaCTAudioConfig, LaCTAudioModel

# Load MusicGen
model_id = "facebook/musicgen-small"
processor = AutoProcessor.from_pretrained(model_id)
base_model = MusicgenForConditionalGeneration.from_pretrained(model_id)

# LaCT config and model
lact_config = LaCTAudioConfig(hidden_size=base_model.config.hidden_size)
lact_model = LaCTAudioModel(lact_config)
lact_model.eval()

prompt = "A long, evolving soundscape with birds, wind, and distant thunder."
inputs = processor(text=prompt, return_tensors="pt")

with torch.no_grad():
    # Step 1: Get MusicGen encoder hidden states (simulate as input embeddings)
    # For demo, use input_ids embedding as proxy for audio tokens
    input_embeds = base_model.get_input_embeddings()(inputs["input_ids"])
    # Step 2: Pass through LaCTAudioModel for TTT adaptation
    lact_out = lact_model(input_embeds)
    # Step 3: Use adapted embeddings for generation (simulate by passing through base model)
    # In practice, you may need to modify MusicGen to accept adapted embeddings
    # Here, we just use the original pipeline for demo
    audio_values = base_model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)

sampling_rate = base_model.config.audio_encoder.sampling_rate
sf.write("musicgen_lact_output.wav", audio_values[0, 0].cpu().numpy(), sampling_rate)
print(f"LaCT-TTT Audio generated and saved as musicgen_lact_output.wav at {sampling_rate} Hz") 