import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, embed_size, feature_size):
        super(LSTM, self).__init__()
        self.relu = nn.ReLU()

        self.lstm_1 = nn.LSTM(input_size=feature_size, hidden_size=embed_size, num_layers=1, batch_first=True, bidirectional=False)
        self.final_out = nn.Linear(embed_size, 3)
        self.softmax = nn.Softmax(dim=2)

        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=2)


    def forward(self, x):
        out, _ = self.lstm_1(x)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.final_out(out)
        out = self.softmax(out)

        return out
