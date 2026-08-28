from __future__ import annotations

from math import cos, pi

import numpy as np
import torch
from einops import einsum, rearrange

Module = torch.nn.Module
Parameter = torch.nn.Parameter
ModuleList = torch.nn.ModuleList
Optimizer = torch.optim.Optimizer


class Linear(Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        self.reset_parameters()
    
    def reset_parameters(self):
        std = (2 / (self.in_features + self.out_features))**0.5
        torch.nn.init.trunc_normal_(self.weight, std=std, a=-3*std, b=3*std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einsum(self.weight, x, "... d_out d_in, ... d_in -> ... d_out")
        return x


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        self.reset_parameters()
    
    def reset_parameters(self):
        std = 1.0
        torch.nn.init.trunc_normal_(self.weight, std=std, a=-3*std, b=3*std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.weight[token_ids, :]
        return x


class RMSNorm(Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = Parameter(torch.empty(d_model, device=device, dtype=dtype))
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.constant_(self.weight, 1.0)

    def forward(self, x):
        dtype = x.dtype
        x = x.to(torch.float32)
        x_norm_inv = 1 / (torch.sqrt(self.eps + (x**2).mean(dim=-1)))
        x = einsum(x, x_norm_inv, self.weight, "... d, ..., d -> ... d")
        x = x.to(dtype)
        return x
    

def silu(x): return x * torch.sigmoid(x)


class SwiGLU(Module):
    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()
        if not d_ff:
            d_ff = d_model * 8 // 3
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x):
        y = self.w1(x)
        y = silu(y)
        x = self.w2(y * self.w3(x))
        return x


class RoPE(Module):
    """Rotary position embeddings over interleaved (even, odd) channel pairs.

    The rotation is stored as two `(max_seq_len, d_k // 2)` cos/sin tables rather
    than `d_k // 2` explicit 2x2 matrices, so a forward pass is a handful of
    elementwise kernels instead of a Python loop over channel pairs.
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k
        pos = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        inv_freq = theta ** (-torch.arange(0, d_k // 2, dtype=torch.float32, device=device) * 2 / d_k)
        angle = torch.outer(pos, inv_freq)               # (max_seq_len, d_k // 2)
        self.register_buffer('cos', torch.cos(angle))
        self.register_buffer('sin', torch.sin(angle))

    def forward(self, x: torch.Tensor, token_positions=None):
        if token_positions is not None:
            cos = self.cos[token_positions]              # (..., seq, d_k // 2)
            sin = self.sin[token_positions]
        else:
            cos = self.cos[:x.size(-2)]                  # (seq, d_k // 2), broadcasts
            sin = self.sin[:x.size(-2)]
        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos
        # Re-interleave: stack pairs on a new trailing axis, then fold it back in.
        return torch.stack((rot_even, rot_odd), dim=-1).flatten(-2)
    
    
# softmax along dimension dim
def softmax(x: torch.Tensor, dim: int):
    x_max = x.max(dim=dim, keepdim=True).values
    x_shifted = x - x_max
    z = torch.exp(x_shifted)
    z_sum = z.sum(dim=dim, keepdim=True)
    z_out = z / z_sum
    return z_out


def attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    X = einsum(Q, K, "... i d_k, ... j d_k -> ... i j") / torch.sqrt(torch.tensor(d_k))
    if mask is not None:
        X = X.masked_fill(~mask, -torch.inf)
    X = softmax(X, -1)
    return einsum(X, V, "... i j, ... j k -> ... i k")


class MultiHeadAttention(Module):
    def __init__(self, d_model, num_heads, theta, max_seq_len, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        d_k = d_model // num_heads
        if theta > 1e-5:
            self.rope = RoPE(theta, d_k, max_seq_len, device=device)
        else:
            self.rope = None
        self.register_buffer('mask', torch.tril(torch.ones((max_seq_len, max_seq_len), dtype=torch.bool, device=device)))
        self.weight = Parameter(torch.empty(3, num_heads * d_k, d_model, device=device, dtype=dtype))
        self.proj = Linear(num_heads * d_k, d_model, device=device, dtype=dtype)
        self.reset_parameters()
    
    def reset_parameters(self):
        std = (2 / (self.weight.size(-1) + self.weight.size(-2)))**0.5
        torch.nn.init.trunc_normal_(self.weight, std=std, a=-3*std, b=3*std)

    def forward(self, x, token_pos):
        seq_len = x.size(-2)
        mask = self.mask[:seq_len, :seq_len]
        x = einsum(self.weight, x, 'qkv i j, ... j -> qkv ... i')
        Q, K, V = x[0, ...], x[1, ...], x[2, ...]
        # Fold the head axis out of the channel dim so every head is attended to
        # in one batched call, instead of looping over heads in Python.
        Q, K, V = (rearrange(t, '... s (h d) -> ... h s d', h=self.num_heads) for t in (Q, K, V))
        if self.rope:
            # token_pos is (..., seq); give it a singleton head axis so the cos/sin
            # tables broadcast across heads rather than against the batch dim.
            pos = token_pos.unsqueeze(-2) if (token_pos is not None and token_pos.dim() > 1) else token_pos
            Q = self.rope(Q, pos)
            K = self.rope(K, pos)
        x = attention(Q, K, V, mask)
        x = rearrange(x, '... h s d -> ... s (h d)')
        return self.proj(x)
        

class TransformerBlock(Module):
    def __init__(self, d_model, num_heads, d_ff, theta, max_seq_len, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.rmsnorm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.rmsnorm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadAttention(d_model, num_heads, theta, max_seq_len, device=device, dtype=dtype)

    def forward(self, x, token_positions):
        y = self.rmsnorm1(x)
        y = self.attn(y, token_positions)
        x = x + y
        y = self.rmsnorm2(x)
        y = self.ffn(y)
        x = x + y
        return x
    

class transformerLM(Module):
    def __init__(self, 
                vocab_size: int,
                context_length: int,
                d_model: int,
                num_layers: int,
                num_heads: int,
                d_ff: int,
                theta: float,
                device=None,
                dtype=None):
        super().__init__()
        self.in_embed = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                TransformerBlock(d_model, num_heads, d_ff, theta, context_length, device=device, dtype=dtype)
            )
        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.out_embed = Linear(d_model, vocab_size, device=device, dtype=dtype)
        
    def forward(self, x, token_pos=None):
        x = self.in_embed(x)
        for layer in self.layers:
            x = layer(x, token_pos)
        x = self.norm(x)
        x = self.out_embed(x)
        # x = softmax(x, dim=-1)
        return x
        

def cross_entropy_loss(x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy over every leading dimension. `x` is left untouched."""
    # Out-of-place: `x` is a live node in the autograd graph, subtracting in
    # place would corrupt the caller's logits.
    x = x - x.max(dim=-1, keepdim=True).values
    logsumexp = torch.log(x.exp().sum(dim=-1))
    x_target = torch.gather(x, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return (logsumexp - x_target).mean()

    
class AdamW(Optimizer):
    def __init__(self, params, lr, betas, weight_decay, eps):
        beta1, beta2 = betas
        lr, beta1, beta2, weight_decay, eps = (torch.tensor(a) for a in (lr, beta1, beta2, weight_decay, eps))
        defaults = {'lr': lr, 'beta1': beta1, 'beta2': beta2, 'gamma': weight_decay, 'eps': eps}
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['m'] = torch.zeros(p.shape, dtype=p.dtype, device=p.device)
                self.state[p]['v'] = torch.zeros(p.shape, dtype=p.dtype, device=p.device)
                self.state[p]['t'] = 1
    
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, beta1, beta2, gamma, eps = (group[k] for k in ['lr', 'beta1', 'beta2', 'gamma', 'eps'])
            for p in group['params']:
                g = p.grad
                m = self.state[p]['m']
                v = self.state[p]['v']
                t = self.state[p]['t']
                
                m *= beta1
                m += (1 - beta1) * g
                v *= beta2
                v += (1 - beta2) * g.pow(2)
                lr_t = lr * (1-beta2.pow(t)).pow(0.5) / (1-beta1.pow(t))
                p -= lr_t * m / (v.pow(0.5) + eps)
                p -= lr * gamma * p

                self.state[p]['t'] += 1


def get_lr_cosine_schedule(t, alpha_max, alpha_min, T_w, T_c):
    if t < T_w:
        return (t / T_w) * alpha_max
    elif t <= T_c:
        progress = (t - T_w) / (T_c - T_w)
        cosine_term = cos(progress * pi)

        return alpha_min + 0.5 * (1 + cosine_term) * (alpha_max - alpha_min)
    else:
        return alpha_min


@torch.no_grad()
def gradient_clipping(params, M, eps=1e-6):
    # Materialise once: `params` is usually a generator, and the old two-pass
    # version silently scaled nothing because the second pass saw an empty one.
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return
    # Accumulate on the gradients' own device; a CPU scalar seed raises here on CUDA.
    total_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))
    if total_norm > M:
        scale = M / (total_norm + eps)
        for g in grads:
            g *= scale
        

def data_loader(seq, context_length, batch_size, device, rng=None):
    """Sample a batch of (input, target) windows uniformly at random.

    Pass `rng` (a np.random.Generator) to make a batch reproducible, e.g. so
    validation is measured on the same windows at every eval step.
    """
    n = len(seq)
    starts = (rng or np.random).integers(0, n - context_length, size=batch_size) \
        if rng is not None else np.random.randint(0, n - context_length, size=batch_size)
    tmp = np.empty(shape=(batch_size, context_length + 1), dtype=np.int64)
    for i, idx in enumerate(starts):
        tmp[i, :] = seq[idx:idx + context_length + 1]
    tmpp = torch.tensor(tmp, device=device)
    return tmpp[:, :-1], tmpp[:, 1:]


def save_checkpoint(model, optimizer, iteration, out):
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'iteration': iteration,
    }, out)


def load_checkpoint(src, model: Module, optimizer: Optimizer):
    d = torch.load(src)
    model.load_state_dict(d['model_state'])
    optimizer.load_state_dict(d['optimizer_state'])
    return d['iteration']