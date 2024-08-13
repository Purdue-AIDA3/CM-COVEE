from Transformer_Block import TransformerBlock
import torch.nn as nn

class Transformer_Encoder(nn.Module):
    def __init__(self, embed_size,
                 num_layers, heads, forward_expansion,
                 dropout, max_length):
        super(Transformer_Encoder, self).__init__()
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.embedding = nn.Linear(max_length, embed_size)
        self.softmax = nn.Softmax(dim=2)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                ) for _ in range(num_layers)
            ]
        )
        self.final_out = nn.Linear(embed_size,3)


    def forward(self, x, pos_encoding, mask, inverse_mask):

        out = self.embedding(x) + pos_encoding # Need to be changed
        for layers in self.layers:
            out = layers(out, out, out, mask, inverse_mask)

        out = self.final_out(out)
        out = self.softmax(out)

        return out



