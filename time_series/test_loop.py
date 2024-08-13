import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import confusion_matrix


def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d / 2)):
            denominator = np.power(n, 2 * i / d)
            P[k, 2 * i] = np.sin(k / denominator)
            P[k, 2 * i + 1] = np.cos(k / denominator)
    return P

def testing(model, model_name, embed_size, test_loader, device):
    model.eval()
    pos_encoding = torch.tensor(
        getPositionEncoding(seq_len=10, d=embed_size, n=10000)).float().to(device)
    attn_shape = (10, 10)
    causal_mask = torch.tensor(np.triu(np.ones(attn_shape), k=1).astype('uint8')).to(device)
    temporal_mask = causal_mask.clone()
    loss = nn.CrossEntropyLoss()

    model.eval()
    all_x = []
    all_y = []
    total = 0
    correct = 0
    with torch.no_grad():
        mse_val = 0
        for batch_idx, (batch_x, batch_y) in enumerate(test_loader):

            if model_name == 'TransformerEncoder':
                out_x = model(batch_x, pos_encoding, causal_mask, temporal_mask)
            elif model_name == 'LSTM':
                out_x = model(batch_x)
            elif model_name == 'GRU':
                out_x = model(batch_x)

            l = loss(out_x.flatten(0, 1), batch_y.flatten(0, 1))
            mse_val += l.item()
            _, predicted = torch.max(out_x.data, 2)
            all_x.append((predicted))
            all_y.append((batch_y))
            total += int(batch_y.size(0) * 10)
            correct += (predicted == batch_y).sum().item()

    all_x = torch.concat(all_x)
    all_y = torch.concat(all_y)
    confusion = confusion_matrix((torch.flatten(all_y)).cpu().detach().numpy(),
                                 (torch.flatten(all_x)).cpu().detach().numpy())
    print(confusion)
    acc_test = correct / total

    print("====================================================================")
    print("Test Accuracy: ", acc_test)
    print("====================================================================")