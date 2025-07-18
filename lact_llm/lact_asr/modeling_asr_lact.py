import torch
import torch.nn as nn
from lact_llm.lact_model.layer_lact_swiglu import LaCTSWIGLULayer
from lact_llm.lact_model.configuration_lact_swiglu import LaCTSWIGLUConfig

class LaCTASREncoder(nn.Module):
    def __init__(self, config: LaCTSWIGLUConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            LaCTSWIGLULayer(
                hidden_size=config.hidden_size,
                num_attn_heads=config.num_attn_heads,
                num_lact_heads=config.num_lact_heads,
                inter_multi=config.inter_multi,
                window_size=config.window_size,
                lact_chunk_size=config.lact_chunk_size,
                qkv_bias=config.qkv_bias,
                attn_qk_norm=config.attn_qk_norm,
                qkv_silu=config.qkv_silu,
                lr_dim=config.lr_dim,
                use_muon=config.use_muon,
                ttt_prenorm=config.ttt_prenorm,
                ttt_nope=config.ttt_nope,
                lr_parameterization=config.lr_parameterization,
                learnable_ttt_scale=config.learnable_ttt_scale,
                rope_theta=config.rope_theta,
                max_position_embeddings=config.max_position_embeddings,
                layer_idx=idx,
                w0_w2_low_rank=config.w0_w2_low_rank,
                use_momentum=config.use_momentum,
                ttt_loss_type=config.ttt_loss_type,
                fw_init_gain=config.fw_init_gain
            ) for idx in range(config.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)
    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x, *_ = layer(x, attention_mask=attn_mask)
        return self.norm(x)

class LaCTASRDecoder(nn.Module):
    def __init__(self, config: LaCTSWIGLUConfig, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            LaCTSWIGLULayer(
                hidden_size=config.hidden_size,
                num_attn_heads=config.num_attn_heads,
                num_lact_heads=config.num_lact_heads,
                inter_multi=config.inter_multi,
                window_size=config.window_size,
                lact_chunk_size=config.lact_chunk_size,
                qkv_bias=config.qkv_bias,
                attn_qk_norm=config.attn_qk_norm,
                qkv_silu=config.qkv_silu,
                lr_dim=config.lr_dim,
                use_muon=config.use_muon,
                ttt_prenorm=config.ttt_prenorm,
                ttt_nope=config.ttt_nope,
                lr_parameterization=config.lr_parameterization,
                learnable_ttt_scale=config.learnable_ttt_scale,
                rope_theta=config.rope_theta,
                max_position_embeddings=config.max_position_embeddings,
                layer_idx=idx,
                w0_w2_low_rank=config.w0_w2_low_rank,
                use_momentum=config.use_momentum,
                ttt_loss_type=config.ttt_loss_type,
                fw_init_gain=config.fw_init_gain
            ) for idx in range(config.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, vocab_size)
    def forward(self, x, attn_mask=None):
        x = self.embedding(x)
        for layer in self.layers:
            x, *_ = layer(x, attention_mask=attn_mask)
        x = self.norm(x)
        return self.lm_head(x)

class LaCTASRModel(nn.Module):
    def __init__(self, config: LaCTSWIGLUConfig, vocab_size: int):
        super().__init__()
        self.encoder = LaCTASREncoder(config)
        self.decoder = LaCTASRDecoder(config, vocab_size)
    def forward(self, audio_features, decoder_input_ids, encoder_mask=None, decoder_mask=None):
        encoder_out = self.encoder(audio_features, attn_mask=encoder_mask)
        logits = self.decoder(decoder_input_ids, attn_mask=decoder_mask)
        return logits 