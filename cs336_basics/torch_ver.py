from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x):
        dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps)
        x = (x / rms) * self.weight
        return x.to(dtype)


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k
        positions = torch.arange(max_seq_len, device=device)
        weight = torch.zeros(d_k // 2, max_seq_len, 2, 2, device=device)
        for k in range(d_k // 2):
            angle = positions / theta ** (2 * k / d_k)
            weight[k, :, 0, 0] = torch.cos(angle)
            weight[k, :, 1, 1] = torch.cos(angle)
            weight[k, :, 0, 1] = -torch.sin(angle)
            weight[k, :, 1, 0] = torch.sin(angle)
        self.register_buffer('weight', weight)

    def forward(self, x: torch.Tensor, token_positions=None):
        tmp = []
        for k in range(self.d_k // 2):
            if token_positions is not None:
                w_k = self.weight[k, token_positions, :, :]
            else:
                w_k = self.weight[k, :x.size(-2), :, :]
            x_k = x[..., 2 * k:2 * k + 2]
            x_k = torch.einsum('...ij,...j->...i', w_k, x_k)
            tmp.append(x_k)
        return torch.cat(tmp, dim=-1)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        if not d_ff:
            d_ff = d_model * 8 // 3
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self._init_weights()

    def _init_weights(self):
        for linear in [self.w1, self.w2, self.w3]:
            std = (2 / (linear.in_features + linear.out_features)) ** 0.5
            nn.init.trunc_normal_(linear.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, theta, max_seq_len, device=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        if theta > 1e-5:
            self.rope = RoPE(theta, self.d_k, max_seq_len, device=device)
        else:
            self.rope = None
        self.q_proj = nn.Linear(d_model, d_model, bias=False, device=device)
        self.k_proj = nn.Linear(d_model, d_model, bias=False, device=device)
        self.v_proj = nn.Linear(d_model, d_model, bias=False, device=device)
        self.proj = nn.Linear(d_model, d_model, bias=False, device=device)
        self._init_weights()

    def _init_weights(self):
        for linear in [self.q_proj, self.k_proj, self.v_proj, self.proj]:
            std = (2 / (linear.in_features + linear.out_features)) ** 0.5
            nn.init.trunc_normal_(linear.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x, token_pos):
        B, L, D = x.shape
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape to (B, num_heads, L, d_k)
        Q = Q.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        if self.rope:
            Q = self.rope(Q, token_pos)
            K = self.rope(K, token_pos)

        x = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

        # Reshape back to (B, L, D)
        x = x.transpose(1, 2).contiguous().view(B, L, D)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta, max_seq_len):
        super().__init__()
        self.ffn = SwiGLU(d_model, d_ff)
        self.rmsnorm1 = RMSNorm(d_model)
        self.rmsnorm2 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, theta, max_seq_len)

    def forward(self, x, token_positions):
        y = self.rmsnorm1(x)
        y = self.attn(y, token_positions)
        x = x + y
        y = self.rmsnorm2(x)
        y = self.ffn(y)
        x = x + y
        return x


class transformerLM_torch(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 theta: float):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, theta, context_length)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.out_embed = nn.Linear(d_model, vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.in_embed.weight, std=1.0, a=-3.0, b=3.0)
        std = (2 / (self.out_embed.in_features + self.out_embed.out_features)) ** 0.5
        nn.init.trunc_normal_(self.out_embed.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x, token_pos=None):
        x = self.in_embed(x)
        for layer in self.layers:
            x = layer(x, token_pos)
        x = self.norm(x)
        x = self.out_embed(x)
        return x
