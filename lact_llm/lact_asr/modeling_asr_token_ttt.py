import torch
import torch.nn as nn
from lact_llm.lact_model.layer_lact_swiglu import LaCTSWIGLULayer
from lact_llm.lact_model.configuration_lact_swiglu import LaCTSWIGLUConfig

class TokenTTTASREncoder(nn.Module):
    def __init__(self, config: LaCTSWIGLUConfig):
        super().__init__()
        # Set chunk size to 1 for per-token TTT
        config_token = LaCTSWIGLUConfig(**{**config.__dict__, 'lact_chunk_size': 1})
        self.layers = nn.ModuleList([
            LaCTSWIGLULayer(
                hidden_size=config_token.hidden_size,
                num_attn_heads=config_token.num_attn_heads,
                num_lact_heads=config_token.num_lact_heads,
                inter_multi=config_token.inter_multi,
                window_size=config_token.window_size,
                lact_chunk_size=1,
                qkv_bias=config_token.qkv_bias,
                attn_qk_norm=config_token.attn_qk_norm,
                qkv_silu=config_token.qkv_silu,
                lr_dim=config_token.lr_dim,
                use_muon=config_token.use_muon,
                ttt_prenorm=config_token.ttt_prenorm,
                ttt_nope=config_token.ttt_nope,
                lr_parameterization=config_token.lr_parameterization,
                learnable_ttt_scale=config_token.learnable_ttt_scale,
                rope_theta=config_token.rope_theta,
                max_position_embeddings=config_token.max_position_embeddings,
                layer_idx=idx,
                w0_w2_low_rank=config_token.w0_w2_low_rank,
                use_momentum=config_token.use_momentum,
                ttt_loss_type=config_token.ttt_loss_type,
                fw_init_gain=config_token.fw_init_gain
            ) for idx in range(config_token.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(config_token.hidden_size)
    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x, *_ = layer(x, attention_mask=attn_mask)
        return self.norm(x)

class TokenTTTASRDecoder(nn.Module):
    def __init__(self, config: LaCTSWIGLUConfig, vocab_size: int):
        super().__init__()
        config_token = LaCTSWIGLUConfig(**{**config.__dict__, 'lact_chunk_size': 1})
        self.embedding = nn.Embedding(vocab_size, config_token.hidden_size)
        self.layers = nn.ModuleList([
            LaCTSWIGLULayer(
                hidden_size=config_token.hidden_size,
                num_attn_heads=config_token.num_attn_heads,
                num_lact_heads=config_token.num_lact_heads,
                inter_multi=config_token.inter_multi,
                window_size=config_token.window_size,
                lact_chunk_size=1,
                qkv_bias=config_token.qkv_bias,
                attn_qk_norm=config_token.attn_qk_norm,
                qkv_silu=config_token.qkv_silu,
                lr_dim=config_token.lr_dim,
                use_muon=config_token.use_muon,
                ttt_prenorm=config_token.ttt_prenorm,
                ttt_nope=config_token.ttt_nope,
                lr_parameterization=config_token.lr_parameterization,
                learnable_ttt_scale=config_token.learnable_ttt_scale,
                rope_theta=config_token.rope_theta,
                max_position_embeddings=config_token.max_position_embeddings,
                layer_idx=idx,
                w0_w2_low_rank=config_token.w0_w2_low_rank,
                use_momentum=config_token.use_momentum,
                ttt_loss_type=config_token.ttt_loss_type,
                fw_init_gain=config_token.fw_init_gain
            ) for idx in range(config_token.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(config_token.hidden_size)
        self.lm_head = nn.Linear(config_token.hidden_size, vocab_size)
    def forward(self, x, attn_mask=None):
        x = self.embedding(x)
        for layer in self.layers:
            x, *_ = layer(x, attention_mask=attn_mask)
        x = self.norm(x)
        return self.lm_head(x)

class TokenTTTASRModel(nn.Module):
    def __init__(self, config: LaCTSWIGLUConfig, vocab_size: int):
        super().__init__()
        self.encoder = TokenTTTASREncoder(config)
        self.decoder = TokenTTTASRDecoder(config, vocab_size)
    def forward(self, audio_features, decoder_input_ids, encoder_mask=None, decoder_mask=None):
        encoder_out = self.encoder(audio_features, attn_mask=encoder_mask)
        logits = self.decoder(decoder_input_ids, attn_mask=decoder_mask)
        return logits 