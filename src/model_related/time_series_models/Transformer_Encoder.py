from Transformer_Block import TransformerBlock
import torch
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

        out = self.embedding(x) + pos_encoding
        for layers in self.layers:
            out = layers(out, out, out, mask, inverse_mask)

        out = self.final_out(out)
        out = self.softmax(out)

        return out


class Transformer_Encoder_Multi(nn.Module):
    def __init__(self, embed_size,
                 num_layers, heads, forward_expansion,
                 dropout, max_length):
        super(Transformer_Encoder_Multi, self).__init__()
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.embedding = nn.Linear(max_length,embed_size)
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

    def forward(self, x, pos_encoding, mask, inverse_mask):

        out = self.embedding(x) + pos_encoding
        for layers in self.layers:
            out = layers(out,out,out,mask,inverse_mask)

        return out


class Aggregater_2(nn.Module):
    def __init__(self, model_1, model_2, embed_size):
        super(Aggregater_2, self).__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        self.final_out = nn.Linear(int(embed_size*2), 3)
        self.softmax = nn.Softmax(dim=2)
        self.relu = nn.ReLU()


    def forward(self, x_1, x_2, pos_encoding, causal_mask, temporal_mask):
        out_1 = self.model_1(x_1, pos_encoding, causal_mask, temporal_mask)
        out_2 = self.model_2(x_2, pos_encoding, causal_mask, temporal_mask)
        out = torch.cat((out_1, out_2), dim=2)
        out = self.final_out(out)

        return out

class Aggregater_3(nn.Module):
    def __init__(self, model_1, model_2, model_3, embed_size):
        super(Aggregater_3, self).__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        self.model_3 = model_3
        self.final_out = nn.Linear(int(embed_size*3), 3)
        self.softmax = nn.Softmax(dim=2)
        self.relu = nn.ReLU()


    def forward(self, x_1, x_2, x_3, pos_encoding, causal_mask, temporal_mask):
        out_1 = self.model_1(x_1, pos_encoding, causal_mask, temporal_mask)
        out_2 = self.model_2(x_2, pos_encoding, causal_mask, temporal_mask)
        out_3 = self.model_3(x_3, pos_encoding, causal_mask, temporal_mask)
        out = torch.cat((out_1, out_2, out_3), dim=2)
        out = self.final_out(out)
        
        return out