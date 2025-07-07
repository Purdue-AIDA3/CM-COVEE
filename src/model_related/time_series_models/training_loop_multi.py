import Transformer_Encoder
import LSTM
import GRU
import torch
import torch.nn as nn
import numpy as np
import time
from sklearn.metrics import confusion_matrix
import pandas as pd
import os

class TrainingLoop:
    def __init__(self, modality, train_loader, val_loader, batchsize, epoch, embed_size, num_layers,
                 head, forward_expansion, dropout, max_length, feature_length, device, lr, model_name, ground_truth):
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
        self.feature_length_1 = feature_length[0]
        self.feature_length_2 = feature_length[1]
        if len(feature_length) == 3:
            self.feature_length_3 = feature_length[2]
        else:
            self.feature_length_3 = []
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

    def training_multi(self):

        def he_init(m):  # He initialization
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if self.model_name == 'TransformerEncoder':
            model_1 = Transformer_Encoder.Transformer_Encoder_Multi(embed_size=self.embed_size, num_layers=self.num_layers, heads=self.head, \
                                        forward_expansion=self.forward_expansion, dropout=self.dropout, max_length=self.feature_length_1).to(self.device)
            model_2 = Transformer_Encoder.Transformer_Encoder_Multi(embed_size=self.embed_size, num_layers=self.num_layers, heads=self.head, \
                                          forward_expansion=self.forward_expansion, dropout=self.dropout,
                                          max_length=self.feature_length_2).to(self.device)
            if self.feature_length_3:
                model_3 = Transformer_Encoder.Transformer_Encoder_Multi(embed_size=self.embed_size, num_layers=self.num_layers, heads=self.head, \
                                                                        forward_expansion=self.forward_expansion, dropout=self.dropout, max_length=self.feature_length_3).to(self.device)
                model = Transformer_Encoder.Aggregater_3(model_1, model_2, model_3, self.embed_size).to(self.device)
            else:
                model = Transformer_Encoder.Aggregater_2(model_1, model_2, self.embed_size).to(self.device)
        elif self.model_name == 'LSTM':
            model_1 = LSTM.LSTM_Multi(embed_size=self.embed_size, feature_size=self.feature_length_1).to(self.device)
            model_2 = LSTM.LSTM_Multi(embed_size=self.embed_size, feature_size=self.feature_length_2).to(self.device)
            if self.feature_length_3:
                model_3 = LSTM.LSTM_Multi(embed_size=self.embed_size, feature_size=self.feature_length_3).to(self.device)
                model = LSTM.Aggregater_3(model_1, model_2, model_3, self.embed_size).to(self.device)
            else:
                model = LSTM.Aggregater_2(model_1, model_2, self.embed_size).to(self.device)
        elif self.model_name == 'GRU':
            model_1 = GRU.GRU_Multi(embed_size=self.embed_size, feature_size=self.feature_length_1).to(self.device)
            model_2 = GRU.GRU_Multi(embed_size=self.embed_size, feature_size=self.feature_length_2).to(self.device)
            if self.feature_length_3:
                model_3 = GRU.GRU_Multi(embed_size=self.embed_size, feature_size=self.feature_length_3).to(self.device)
                model = GRU.Aggregater_3(model_1, model_2, model_3, self.embed_size).to(self.device)
            else:
                model = GRU.Aggregater_2(model_1, model_2, self.embed_size).to(self.device)

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

            if self.feature_length_3:
                for batch_idx, (batch_x_1, batch_x_2, batch_x_3, batch_y) in enumerate(self.train_loader):

                    if self.model_name == 'TransformerEncoder':
                        out_x = model(batch_x_1, batch_x_2, batch_x_3, pos_encoding, causal_mask, temporal_mask)
                    elif self.model_name == 'LSTM':
                        out_x = model(batch_x_1, batch_x_2, batch_x_3)
                    elif self.model_name == 'GRU':
                        out_x = model(batch_x_1, batch_x_2, batch_x_3)

                    l = loss(out_x.flatten(0, 1), batch_y.flatten(0, 1))
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
            else:
                for batch_idx, (batch_x_1, batch_x_2, batch_y) in enumerate(self.train_loader):

                    if self.model_name == 'TransformerEncoder':
                        out_x = model(batch_x_1, batch_x_2, pos_encoding, causal_mask, temporal_mask)
                    elif self.model_name == 'LSTM':
                        out_x = model(batch_x_1, batch_x_2)
                    elif self.model_name == 'GRU':
                        out_x = model(batch_x_1, batch_x_2)

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
            all_train_acc.append(acc_train)
            y_true = torch.concat(y_true)
            y_pred = torch.concat(y_pred)
            confusion = confusion_matrix((torch.flatten(y_true)).cpu().detach().numpy(),
                                         (torch.flatten(y_pred)).cpu().detach().numpy())
            print(confusion)
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
                if self.feature_length_3:
                    for batch_idx, (batch_x_1, batch_x_2, batch_x_3, batch_y) in enumerate(self.val_loader):
                        if self.model_name == 'TransformerEncoder':
                            out_x = model(batch_x_1, batch_x_2, batch_x_3, pos_encoding, causal_mask, temporal_mask)
                        elif self.model_name == 'LSTM':
                            out_x = model(batch_x_1, batch_x_2, batch_x_3)
                        elif self.model_name == 'GRU':
                            out_x = model(batch_x_1, batch_x_2, batch_x_3)

                        l = loss(out_x.flatten(0,1), batch_y.flatten(0,1))
                        mse_val += l.item()
                        _, predicted = torch.max(out_x.data, 2)
                        total += int(batch_y.size(0) * 10)
                        correct += (predicted == batch_y).sum().item()
                        all_x.append((predicted))
                        all_y.append((batch_y))
                else:
                    for batch_idx, (batch_x_1, batch_x_2, batch_y) in enumerate(self.val_loader):
                        if self.model_name == 'TransformerEncoder':
                            out_x = model(batch_x_1, batch_x_2, pos_encoding, causal_mask, temporal_mask)
                        elif self.model_name == 'LSTM':
                            out_x = model(batch_x_1, batch_x_2)
                        elif self.model_name == 'GRU':
                            out_x = model(batch_x_1, batch_x_2)

                        l = loss(out_x.flatten(0, 1), batch_y.flatten(0, 1))
                        mse_val += l.item()
                        _, predicted = torch.max(out_x.data, 2)
                        total += int(batch_y.size(0) * 10)
                        correct += (predicted == batch_y).sum().item()
                        all_x.append((predicted))
                        all_y.append((batch_y))

            mse_val = mse_val / (num_val_batch*10)
            all_val_mse.append(mse_val)
            all_x = torch.concat(all_x)
            all_y = torch.concat(all_y)
            acc_val = correct / total
            all_val_acc.append(acc_val)
            confusion = confusion_matrix((torch.flatten(all_y)).cpu().detach().numpy(),
                                         (torch.flatten(all_x)).cpu().detach().numpy())
            print(confusion)

            print("====================================================================")
            print("Training Loss: ", mse_train, "      Val Loss: ", mse_val)
            print("Training Accuracy: ", acc_train, "      Val Accuracy: ", acc_val)
            print("====================================================================")

            # data = {
            #     'Epoch': [i],
            #     'All Training Accuracy': [acc_train],
            #     'All Testing Accuracy': [acc_val],
            #     'Confusion Matrix': [confusion.flatten()]
            # }
            # new_df = pd.DataFrame(data)
            # new_df['Confusion Matrix'] = new_df['Confusion Matrix'].apply(lambda x: str(x))
            # new_df.to_csv('output.csv', index=False)
            # if os.path.isfile(
            #         'covee_src/' + self.ground_truth + '/' + self.model_name + '/' + self.modality + '_test_outputs.csv'):
            #     # new_df = pd.DataFrame(data)
            #     new_df.to_csv('covee_src/' + self.ground_truth + '/' + self.model_name + '/' + self.modality + '_test_outputs.csv',
            #                   mode='a',
            #                   header=False, index=False)
            # else:
            #     new_df.to_csv('covee_src/' + self.ground_truth + '/' + self.model_name + '/' + self.modality + '_test_outputs.csv',
            #                   index=False)

        return model, all_train_mse, all_val_mse, all_train_acc, all_val_acc

    def testing_multi(self, model, test_loader):
        model.eval()
        all_x = []
        all_y = []
        pos_encoding = torch.tensor(self.getPositionEncoding(seq_len=self.max_length, d=self.embed_size, n=10000)).float().to(self.device)
        attn_shape = (self.max_length, self.max_length)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        causal_mask = (torch.from_numpy(subsequent_mask) == 0).to(self.device)
        temporal_mask = causal_mask.clone()
        loss = nn.CrossEntropyLoss()
        num_test_batch = len(test_loader)
        total = 0
        correct = 0

        with torch.no_grad():
            mse_test = 0

            if self.feature_length_3:
                for batch_idx, (batch_x_1, batch_x_2, batch_x_3, batch_y) in enumerate(test_loader):

                    if self.model_name == 'TransformerEncoder':
                        out_x = model(batch_x_1, batch_x_2, batch_x_3, pos_encoding, causal_mask, temporal_mask)
                    elif self.model_name == 'LSTM':
                        out_x = model(batch_x_1, batch_x_2, batch_x_3)
                    elif self.model_name == 'GRU':
                        out_x = model(batch_x_1, batch_x_2, batch_x_3)

                    l = loss(out_x.flatten(0, 1), batch_y.flatten(0, 1))
                    mse_test += l.item()

                    _, predicted = torch.max(out_x.data, 2)
                    total += int(batch_y.size(0) * 10)
                    correct += (predicted == batch_y).sum().item()
                    all_x.append((predicted))
                    all_y.append((batch_y))
            else:
                for batch_idx, (batch_x_1, batch_x_2, batch_y) in enumerate(test_loader):

                    if self.model_name == 'TransformerEncoder':
                        out_x = model(batch_x_1, batch_x_2, pos_encoding, causal_mask, temporal_mask)
                    elif self.model_name == 'LSTM':
                        out_x = model(batch_x_1, batch_x_2)
                    elif self.model_name == 'GRU':
                        out_x = model(batch_x_1, batch_x_2)

                    l = loss(out_x.flatten(0, 1), batch_y.flatten(0, 1))
                    mse_test += l.item()

                    _, predicted = torch.max(out_x.data, 2)
                    total += int(batch_y.size(0) * 10)
                    correct += (predicted == batch_y).sum().item()
                    all_x.append((predicted))
                    all_y.append((batch_y))

        mse_test = mse_test / (num_test_batch * 10)
        all_x = torch.concat(all_x)
        all_y = torch.concat(all_y)
        acc = correct / total

        print("====================================================================")
        print("Test Acc: ", acc)
        print("====================================================================")
        return all_x, all_y, acc