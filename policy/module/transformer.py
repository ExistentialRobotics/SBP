from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import xformers.ops as xops

from policy.module.mlp import init_weights_kaiming


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        use_xformers: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_xformers = use_xformers

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.activation = F.gelu

    def forward(
        self,
        src: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, _ = src.shape

        # All projected to (B, S, H, D)
        q = self.W_q(src).view(B, S, self.n_heads, self.head_dim)
        k = self.W_k(src).view(B, S, self.n_heads, self.head_dim)
        v = self.W_v(src).view(B, S, self.n_heads, self.head_dim)

        if self.use_xformers:
            # xformers expects (B, S, H, D)
            attn_bias = None
            if key_padding_mask is not None:
                seq_len = key_padding_mask.size(1)
                mask = key_padding_mask[:, None, None, :].to(q.dtype)
                attn_bias = mask.expand(-1, self.n_heads, seq_len, -1) * (-1e9)

            attn = xops.memory_efficient_attention(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                attn_bias=attn_bias,
                p=self.dropout_attn.p if self.training else 0.0,
            )  # (B, S, H, D)

        else:
            # sdpa expects (B, H, S, D)
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()

            attn_mask = None
            if key_padding_mask is not None:
                # sdpa: True = allow attend, False = mask out
                # key_padding_mask: True = pad → invert
                attn_mask = ~key_padding_mask[:, None, None, :].to(torch.bool)

            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout_attn.p if self.training else 0.0,
            )
            attn = attn.transpose(1, 2).contiguous()  # (B, S, H, D)

        attn = attn.reshape(B, S, self.d_model).contiguous()

        src2 = self.norm1(src + self.dropout_attn(self.out_proj(attn)))
        ff = self.linear2(self.activation(self.linear1(src2)))
        out = self.norm2(src2 + self.dropout_ff(ff))
        return out


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        num_heads: int = 8,
        max_seq_len: int = 1024,
    ):
        super().__init__()

        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, input_dim))

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=input_dim,
                    n_heads=num_heads,
                    dim_feedforward=hidden_dim,
                    use_xformers=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.apply(init_weights_kaiming)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(
        self,
        visual_token: torch.Tensor,
        use_pe: bool = True,
    ) -> torch.Tensor:

        S = visual_token.shape[1]
        if use_pe:
            visual_token = visual_token + self.pos_embed[:, :S]

        for layer in self.layers:
            visual_token = layer(src=visual_token)

        return visual_token


class XformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        self.self_attn_proj = nn.Linear(d_model, 3 * d_model)
        self.self_attn_out = nn.Linear(d_model, d_model)

        self.cross_attn_q_proj = nn.Linear(d_model, d_model)
        self.cross_attn_kv_proj = nn.Linear(d_model, 2 * d_model)
        self.cross_attn_out = nn.Linear(d_model, d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout_attn1 = nn.Dropout(dropout)
        self.dropout_attn2 = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

        self.activation = F.gelu

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask_bias=None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = tgt.shape

        # Masked Self-Attention
        q, k, v = self.self_attn_proj(tgt).chunk(3, dim=-1)
        q = q.view(B, T, self.nhead, D // self.nhead)
        k = k.view(B, T, self.nhead, D // self.nhead)
        v = v.view(B, T, self.nhead, D // self.nhead)

        self_attn_out = xops.memory_efficient_attention(q, k, v, attn_bias=tgt_mask_bias)
        self_attn_out = self_attn_out.view(B, T, D)

        tgt = self.norm1(tgt + self.dropout_attn1(self.self_attn_out(self_attn_out)))

        # Cross-Attention
        q = self.cross_attn_q_proj(tgt)
        k, v = self.cross_attn_kv_proj(memory).chunk(2, dim=-1)

        _, S, _ = memory.shape

        q = q.view(B, T, self.nhead, D // self.nhead)
        k = k.view(B, S, self.nhead, D // self.nhead)
        v = v.view(B, S, self.nhead, D // self.nhead)

        cross_attn_bias = None
        if memory_key_padding_mask is not None:
            mask = memory_key_padding_mask[:, None, None, :].to(q.dtype)
            cross_attn_bias = mask.expand(-1, self.nhead, T, -1) * (-1e9)

        cross_attn_out = xops.memory_efficient_attention(q, k, v, attn_bias=cross_attn_bias)
        cross_attn_out = cross_attn_out.view(B, T, D)

        tgt2 = self.norm2(tgt + self.dropout_attn2(self.cross_attn_out(cross_attn_out)))

        # FFN
        ff_out = self.linear2(self.activation(self.linear1(tgt2)))
        tgt3 = self.norm3(tgt2 + self.dropout_ff(ff_out))

        return tgt3


class ActionTransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        transf_input_dim: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        action_dim: int,
        action_pred_horizon: int = 16,
    ):
        super().__init__()

        self.query_embed = nn.Embedding(action_pred_horizon, d_model)

        self.layers = nn.ModuleList(
            [
                XformerDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        self.action_head = nn.Linear(d_model, action_dim)
        self.action_pred_horizon = action_pred_horizon

        self.memory_proj = nn.Linear(transf_input_dim, d_model)
        self.apply(init_weights_kaiming)

        self.causal_attn_bias = xops.LowerTriangularMask()

    def forward(
        self,
        visual_token: torch.Tensor,
        state_tok: torch.Tensor,
        text_emb: Optional[torch.Tensor],
        global_tok: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, _, d_model = visual_token.shape

        # Build memory and padding mask
        memory_parts = [visual_token, state_tok]
        padding_masks = [
            torch.zeros((B, visual_token.shape[1]), device=visual_token.device, dtype=torch.bool),
            torch.zeros((B, 1), device=state_tok.device, dtype=torch.bool),
        ]

        if text_emb is not None:
            memory_parts.append(text_emb)
            padding_masks.append(torch.zeros((B, 1), device=text_emb.device, dtype=torch.bool))

        if global_tok is not None:
            memory_parts.append(global_tok)
            padding_masks.append(torch.zeros((B, 1), device=global_tok.device, dtype=torch.bool))

        tokens = torch.cat(memory_parts, dim=1)
        tokens = self.memory_proj(tokens)
        memory_key_padding_mask = torch.cat(padding_masks, dim=1)

        # Pad memory to a multiple of 8 for xformers efficiency
        S = tokens.shape[1]
        pad_to = (S + 7) & (-8)
        if S < pad_to:
            pad_len = pad_to - S
            tokens = F.pad(tokens, (0, 0, 0, pad_len), "constant", 0)
            mask_pad = torch.ones(B, pad_len, device=tokens.device, dtype=torch.bool)
            memory_key_padding_mask = torch.cat([memory_key_padding_mask, mask_pad], dim=1)

        query_pos = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)

        decoder_out = query_pos
        for layer in self.layers:
            decoder_out = layer(
                tgt=decoder_out,
                memory=tokens,
                tgt_mask_bias=self.causal_attn_bias,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        action_out = self.action_head(decoder_out)
        return action_out
