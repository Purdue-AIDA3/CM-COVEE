import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert(self.head_dim*heads == embed_size) # ensure they're divisible

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.query = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(self.heads*self.head_dim, embed_size)

    def forward(self, values, keys, query, mask, inverse_mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)

        values = self.values(values)
        query = self.query(query)
        keys = self.keys(keys)

        energy = torch.einsum("nqhd,nkhd->nhqk",[query, keys])
        # # query shape: (N, query_len, heads, heads_dim), query_len, key_len = MaxLength
        # # key shape: (N, key_len, heads, heads_dim)
        # # energy shape: (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 1, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])
        out = out.reshape(N, query_len, self.heads*self.head_dim)
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, head_dim), key_len always equal to value_len
        # out shape: (N, query_len, heads, head_dim), then flatten last two dimensions

        out = self.fc_out(out)

        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask, inverse_mask):
        attention = self.attention(value, key, query, mask, inverse_mask)

        x = self.dropout(self.norm1(attention + query)) # Add & Norm
        forward = self.feed_forward(x) # Feed Forward
        out = self.dropout(self.norm2(forward + x)) # Add & Norm

        return out

