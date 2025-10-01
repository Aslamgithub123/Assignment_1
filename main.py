import torch
import torch.nn as nn
import math

# ========================================
# VOCABULARY
# ========================================

# Programming vocabulary (input)
src_vocab = {
    '<pad>': 0,
    '<sos>': 1,  # Start of sequence
    '<eos>': 2,  # End of sequence
    'def': 3,
    'add': 4,
    'numbers': 5,
    'return': 6,
    'sum': 7,
}

# Documentation vocabulary (target)
tgt_vocab = {
    '<pad>': 0,
    '<sos>': 1,
    '<eos>': 2,
    'function': 3,
    'adds': 4,
    'and': 5,
    'returns': 6,
    'total': 7,
}

# Reverse mappings (ID to word)
src_vocab_reverse = {v: k for k, v in src_vocab.items()}
tgt_vocab_reverse = {v: k for k, v in tgt_vocab.items()}


# ========================================
# PART 1: HELPER COMPONENTS
# ========================================

class PositionalEncoding(nn.Module):
    """
    This adds 'position information' to your tokens.
    Think of it like adding timestamps - it helps the model
    know that word 1 comes before word 2.
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    """
    This is the 'attention mechanism' - it helps the model
    focus on important parts of the input.
    """

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = Q.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_head).transpose(1,
                                                                          2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(
            self.d_head)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.softmax(scores, dim=-1)

        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            self.d_model)

        output = self.W_o(context)
        return output, attention_weights


class FeedForward(nn.Module):
    """
    Simple neural network that processes each position.
    """

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.linear1(x))
        x2 = self.linear2(x1)
        return x2


# ========================================
# PART 2: ENCODER BLOCK
# ========================================

class EncoderBlock(nn.Module):
    """
    One encoder layer: Self-Attention → Add & Norm → Feed-Forward → Add & Norm
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.self_attention(x, x, x)
        residual_1 = x + self.dropout(attn_output)
        x = self.norm1(residual_1)
        ff_input = x
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


# ========================================
# PART 3: DECODER BLOCK
# ========================================

class DecoderBlock(nn.Module):
    """
    One decoder layer: Masked Self-Attention → Cross-Attention → Feed-Forward
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.masked_self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        attn_output, _ = self.masked_self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        cross_attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        ff_input = x
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


# ========================================
# PART 4: COMPLETE TRANSFORMER
# ========================================

class SimpleTransformer(nn.Module):
    """
    Complete Transformer: Input → Encoder → Decoder → Output
    """

    def __init__(self, vocab_size, d_model, num_heads, num_encoder_layers,
                 num_decoder_layers, d_ff, max_seq_len, dropout=0.1):
        super().__init__()

        self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        self.encoder_layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def create_masks(self, src, tgt):
        tgt_len = tgt.size(1)
        tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len)).unsqueeze(0).unsqueeze(0)
        return None, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.create_masks(src, tgt)

        embedding_weight = self.encoder_embedding.weight
        src_embedded = self.encoder_embedding(src)
        tgt_embedded = self.decoder_embedding(tgt)

        src_embedded = self.pos_encoding(src_embedded)
        tgt_embedded = self.pos_encoding(tgt_embedded)

        encoder_output = src_embedded
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output)

        decoder_output = tgt_embedded
        for decoder_layer in self.decoder_layers:
            decoder_output = decoder_layer(decoder_output, encoder_output, src_mask, tgt_mask)

        final_decoder = decoder_output
        logits = self.output_projection(decoder_output)
        return logits


# ========================================
# PART 5: RUN THE MODEL
# ========================================

def main():
    print("-" * 70)
    print("TRANSFORMER DEBUGGER")
    print("Input: 'def add numbers return sum' (Programming)")
    print("Target: 'function adds and returns total' (Documentation)")
    print("-" * 70)

    vocab_size = 50  # Increased for real words
    d_model = 128
    num_heads = 4
    num_encoder_layers = 2
    num_decoder_layers = 2
    d_ff = 512
    max_seq_len = 20

    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len
    )

    model.eval()

    # Input: <sos> def add numbers return sum <eos>
    # Token IDs: [1, 3, 4, 5, 6, 7, 2]
    src = torch.tensor([[1, 3, 4, 5, 6, 7, 2]])
    print(f"\nInput tokens: {src.tolist()[0]}")
    print(f"Input words: {[src_vocab_reverse[id] for id in src.tolist()[0]]}")

    # Target: <sos> function adds and returns total <eos>
    # Token IDs: [1, 3, 4, 5, 6, 7, 2]
    tgt = torch.tensor([[1, 3, 4, 5, 6, 7, 2]])
    print(f"\nTarget tokens: {tgt.tolist()[0]}")
    print(f"Target words: {[tgt_vocab_reverse[id] for id in tgt.tolist()[0]]}")

    with torch.no_grad():
        output = model(src, tgt)

    logits_slice = output[0, 0, :10]
    print(f"\n{'' * 70}")
    print(f"Model executed successfully!")
    print(f"Output shape: {output.shape}")
    print(f"First token logits (first 10): {output[0, 0, :10]}")
    print(f"{'-' * 70}")

if __name__ == "__main__":
    main()
