import torch
from assign.components import transformerLM

def test_lm():
    vocab_size = 100
    context_length = 20
    d_model = 32
    num_layers = 2
    num_heads = 4
    d_ff = 64
    rope_theta = 10000.0
    
    lm = transformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)
    
    batch_size = 2
    seq_len = 10
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Check forward pass
    token_pos = torch.arange(seq_len)
    out = lm(x, token_pos)
    
    assert out.shape == (batch_size, seq_len, vocab_size)
    print("LM test passed!")

if __name__ == "__main__":
    test_lm()
