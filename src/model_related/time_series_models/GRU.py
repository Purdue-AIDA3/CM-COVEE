import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, embed_size, feature_size):
        super(GRU, self).__init__()
        self.relu = nn.ReLU()

        self.gru_1_1 = nn.GRU(input_size=feature_size, hidden_size=embed_size, num_layers=1, batch_first=True)
        self.final_out = nn.Linear(embed_size, 3)
        self.softmax = nn.Softmax(dim=2)

        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=2)


    def forward(self, x):
        out, _ = self.gru_1_1(x)
        out = self.relu(out)

        out = self.dropout(out)

        out = self.final_out(out)
        out = self.softmax(out)

        return out

class GRU_Multi(nn.Module):
    def __init__(self, embed_size, feature_size):
        super(GRU_Multi, self).__init__()
        self.relu = nn.ReLU()
        self.gru_1_1 = nn.GRU(input_size=feature_size, hidden_size=embed_size, num_layers=1, batch_first=True)

    def forward(self, x):
        out, _ = self.gru_1_1(x)
        out = self.relu(out)

        return out

class Aggregater_2(nn.Module):
    def __init__(self, model_1, model_2, embed_size):
        super(Aggregater_2, self).__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        self.relu = nn.ReLU()

        self.inter_layer = nn.Linear(int(embed_size * 2), embed_size)
        self.final_out = nn.Linear(embed_size, 3)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()


    def forward(self, x_1, x_2):
        out_1 = self.model_1(x_1)
        out_2 = self.model_2(x_2)
        out = torch.cat((out_1, out_2), dim=2)
        out = self.relu(self.inter_layer(out))
        out = self.dropout(out)
        out = self.final_out(out)

        return out
