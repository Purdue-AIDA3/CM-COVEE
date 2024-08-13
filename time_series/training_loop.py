from Transformer_Encoder import Transformer_Encoder
from LSTM import LSTM
from GRU import GRU
import torch
import torch.nn as nn
import numpy as np
import time
from sklearn.metrics import confusion_matrix

class TrainingLoop:
    def __init__(self, modality, train_loader, val_loader, batchsize, epoch, embed_size, num_layers, head, forward_expansion, dropout, max_length, feature_length, device, lr, model_name, ground_truth):
        self.modality = modality
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batchsize = batchsize
        self.epoch = epoch
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.head = head
        self.forward_expansion = forward_expansion
        self.dropout = dropout
        self.max_length = max_length
        self.feature_length = feature_length
        self.device = device
        self.lr = lr
        self.model_name = model_name
        self.ground_truth = ground_truth

    def getPositionEncoding(self, seq_len, d, n=10000):
        P = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in np.arange(int(d / 2)):
                denominator = np.power(n, 2 * i / d)
                P[k, 2 * i] = np.sin(k / denominator)
                P[k, 2 * i + 1] = np.cos(k / denominator)
        return P

    def training(self):

        def he_init(m):  # He initialization
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if self.model_name == 'TransformerEncoder':
            model = Transformer_Encoder(embed_size=self.embed_size, num_layers=self.num_layers, heads=self.head, \
                                        forward_expansion=self.forward_expansion, dropout=self.dropout, max_length=self.feature_length).to(self.device)
        elif self.model_name == 'LSTM':
            model = LSTM(embed_size=self.embed_size, feature_size=self.feature_length)
        elif self.model_name == 'GRU':
            model = GRU(embed_size=self.embed_size, feature_size=self.feature_length)

        opt = torch.optim.Adam(model.parameters(), lr=self.lr)  # 2e-5
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epoch)
        model.apply(he_init)

        loss = nn.CrossEntropyLoss()
        num_train_batch = len(self.train_loader)
        num_val_batch = len(self.val_loader)
        all_train_mse = []
        all_val_mse = []
        all_train_acc = []
        all_val_acc = []
        pos_encoding = torch.tensor(self.getPositionEncoding(seq_len=self.max_length, d=self.embed_size, n=10000)).float().to(self.device)
        attn_shape = (self.max_length, self.max_length)
        causal_mask = torch.tensor(np.triu(np.ones(attn_shape), k=1).astype('uint8')).to(self.device)
        temporal_mask = causal_mask.clone()

        for i in range(0, self.epoch):
            start_time = time.time()
            mse_train = 0
            model.train()
            total = 0
            correct = 0
            y_true = []
            y_pred = []

            for batch_idx, (batch_x, batch_y) in enumerate(self.train_loader):
                if self.model_name == 'TransformerEncoder':
                    out_x = model(batch_x, pos_encoding, causal_mask, temporal_mask)
                elif self.model_name == 'LSTM':
                    out_x = model(batch_x)
                elif self.model_name == 'GRU':
                    out_x = model(batch_x)

                l = loss(out_x.flatten(0,1), batch_y.flatten(0,1))
                mse_train += l.item()
                opt.zero_grad()
                l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

                _, predicted = torch.max(out_x.data, 2)
                y_true.append((predicted))
                y_pred.append((batch_y))
                total += int(batch_y.size(0) * 10)
                correct += (predicted == batch_y).sum().item()


            scheduler.step()
            mse_train = mse_train / (num_train_batch*10)
            all_train_mse.append(mse_train)
            acc_train = correct / total
            y_true = torch.concat(y_true)
            y_pred = torch.concat(y_pred)
            confusion = confusion_matrix((torch.flatten(y_true)).cpu().detach().numpy(),
                                         (torch.flatten(y_pred)).cpu().detach().numpy())
            print(confusion)
            all_train_acc.append(acc_train)
            print('------------------Done %s epoch------------------' % (i + 1))
            print("--- %s seconds ---" % (time.time() - start_time))
            start_time = time.time()

            model.eval()
            all_x = []
            all_y = []
            total = 0
            correct = 0
            with torch.no_grad():
                mse_val = 0
                for batch_idx, (batch_x, batch_y) in enumerate(self.val_loader):

                    if self.model_name == 'TransformerEncoder':
                        out_x = model(batch_x, pos_encoding, causal_mask, temporal_mask)
                    elif self.model_name == 'LSTM':
                        out_x = model(batch_x)
                    elif self.model_name == 'GRU':
                        out_x = model(batch_x)

                    l = loss(out_x.flatten(0,1), batch_y.flatten(0,1))
                    mse_val += l.item()
                    _, predicted = torch.max(out_x.data, 2)
                    all_x.append((predicted))
                    all_y.append((batch_y))
                    total += int(batch_y.size(0) * 10)
                    correct += (predicted == batch_y).sum().item()

            mse_val = mse_val / (num_val_batch*10)
            all_val_mse.append(mse_val)
            all_x = torch.concat(all_x)
            all_y = torch.concat(all_y)
            confusion = confusion_matrix((torch.flatten(all_y)).cpu().detach().numpy(),
                                         (torch.flatten(all_x)).cpu().detach().numpy())
            print(confusion)
            acc_val = correct / total
            all_val_acc.append(acc_val)

            print("====================================================================")
            print("Training Loss: ", mse_train, "      Val Loss: ", mse_val)
            print("Training Accuracy: ", acc_train, "      Val Accuracy: ", acc_val)
            print("====================================================================")

        return model, all_train_mse, all_val_mse, all_train_acc, all_val_acc
