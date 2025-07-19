import torch
from lact_llm.lact_asr.data import get_librispeech_loader
from lact_llm.lact_asr.utils import LogMelFeatureExtractor, compute_wer, compute_cer, log_hardware_metrics
from lact_llm.lact_model.configuration_lact_swiglu import LaCTSWIGLUConfig
from lact_llm.lact_asr.modeling_asr_lact import LaCTASRModel
from lact_llm.lact_asr.modeling_asr_token_ttt import TokenTTTASRModel
import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Dummy baseline model (no TTT): just a linear projection for demo
class BaselineASRModel(torch.nn.Module):
    def __init__(self, input_dim, vocab_size):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, vocab_size)
    def forward(self, features, decoder_input_ids=None, **kwargs):
        # Ignore decoder_input_ids for baseline
        return self.linear(features)

def decode_logits(logits, vocab):
    # Greedy decode: argmax and map to vocab
    indices = logits.argmax(-1)
    return ["".join([vocab[i] for i in seq if i < len(vocab)]) for seq in indices]

def main():
    # Config
    root = "./data"
    batch_size = 1
    vocab = [chr(i) for i in range(97, 123)] + [" "]  # a-z and space
    vocab_size = len(vocab)
    feature_extractor = LogMelFeatureExtractor()
    loader = get_librispeech_loader(root, url="test-clean", batch_size=batch_size)
    # Model configs
    config = LaCTSWIGLUConfig(hidden_size=80, num_hidden_layers=2, num_attn_heads=4, num_lact_heads=2, inter_multi=2, lact_chunk_size=16)
    # Models
    baseline = BaselineASRModel(input_dim=80, vocab_size=vocab_size)
    chunk_ttt = LaCTASRModel(config, vocab_size)
    token_ttt = TokenTTTASRModel(config, vocab_size)
    # Wav2Vec2 Baseline
    w2v2_model_id = "facebook/wav2vec2-base-960h"
    w2v2_processor = Wav2Vec2Processor.from_pretrained(w2v2_model_id)
    w2v2_model = Wav2Vec2ForCTC.from_pretrained(w2v2_model_id)
    w2v2_model.eval()
    # Results storage
    results = []
    # Evaluation loop (small batch for demo)
    for i, (waveform, sample_rate, transcript) in enumerate(loader):
        if i >= 5: break  # Only a few samples for demo
        features = feature_extractor(waveform[0])  # (time, mel)
        features = features.unsqueeze(0)  # (1, time, mel)
        # Baseline
        logits_base = baseline(features)
        hyp_base = decode_logits(logits_base, vocab)[0]
        # Chunk TTT
        logits_chunk = chunk_ttt(features, decoder_input_ids=None)
        hyp_chunk = decode_logits(logits_chunk, vocab)[0]
        # Token TTT
        logits_token = token_ttt(features, decoder_input_ids=None)
        hyp_token = decode_logits(logits_token, vocab)[0]
        # Wav2Vec2 Baseline
        input_wav = waveform[0].squeeze().numpy()
        w2v2_inputs = w2v2_processor(input_wav, sampling_rate=sample_rate[0], return_tensors="pt", padding=True)
        with torch.no_grad():
            w2v2_logits = w2v2_model(w2v2_inputs.input_values).logits
        w2v2_pred_ids = torch.argmax(w2v2_logits, dim=-1)
        hyp_w2v2 = w2v2_processor.batch_decode(w2v2_pred_ids)[0].lower()
        # Metrics
        ref = transcript[0].lower()
        wer_base = compute_wer(ref, hyp_base)
        wer_chunk = compute_wer(ref, hyp_chunk)
        wer_token = compute_wer(ref, hyp_token)
        wer_w2v2 = compute_wer(ref, hyp_w2v2)
        cer_base = compute_cer(ref, hyp_base)
        cer_chunk = compute_cer(ref, hyp_chunk)
        cer_token = compute_cer(ref, hyp_token)
        cer_w2v2 = compute_cer(ref, hyp_w2v2)
        print(f"Sample {i+1}")
        print(f"REF: {ref}")
        print(f"BASE: {hyp_base}")
        print(f"CHUNK_TTT: {hyp_chunk}")
        print(f"TOKEN_TTT: {hyp_token}")
        print(f"WAV2VEC2: {hyp_w2v2}")
        print(f"WER (BASE): {wer_base:.3f}")
        print(f"WER (CHUNK_TTT): {wer_chunk:.3f}")
        print(f"WER (TOKEN_TTT): {wer_token:.3f}")
        print(f"WER (WAV2VEC2): {wer_w2v2:.3f}")
        print(f"CER (BASE): {cer_base:.3f}")
        print(f"CER (CHUNK_TTT): {cer_chunk:.3f}")
        print(f"CER (TOKEN_TTT): {cer_token:.3f}")
        print(f"CER (WAV2VEC2): {cer_w2v2:.3f}")
        log_hardware_metrics()
        print("-"*40)
        results.append({
            "Sample": i+1,
            "REF": ref,
            "BASE": hyp_base,
            "CHUNK_TTT": hyp_chunk,
            "TOKEN_TTT": hyp_token,
            "WAV2VEC2": hyp_w2v2,
            "WER_BASE": wer_base,
            "WER_CHUNK_TTT": wer_chunk,
            "WER_TOKEN_TTT": wer_token,
            "WER_WAV2VEC2": wer_w2v2,
            "CER_BASE": cer_base,
            "CER_CHUNK_TTT": cer_chunk,
            "CER_TOKEN_TTT": cer_token,
            "CER_WAV2VEC2": cer_w2v2
        })
    # Save results for plotting
    df = pd.DataFrame(results)
    df.to_csv("results_asr_experiment.csv", index=False)
    print("Results saved to results_asr_experiment.csv")

if __name__ == "__main__":
    main() 