## general imports
import random
import math
import os
import logging
import argparse

## numpy
import numpy as np

## pytorch
import torch
import torch.nn as nn
import torch.optim
import torch.optim.lr_scheduler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

## einops
from einops import rearrange

## scikit-learn
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix

## hugging_face 
# from transformers import VivitModel, VivitConfig

import torchvision

## set seeds
random.seed(8000)
torch.manual_seed(8000)
np.random.seed(8000)
rng = np.random.RandomState(seed=8000)


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def he_init(model):  # He initialization
    if isinstance(model, nn.Linear):
        nn.init.kaiming_uniform_(model.weight, nonlinearity='relu')
        if model.bias is not None:
            nn.init.zeros_(model.bias)


class createDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        data = np.load(self.paths[index][:-1])
        eeg = data["eeg"]
        eye = data["eye"]
        face = data["face"]
        video = data["video"]
        label = data["label"]
        video = rearrange(video, 'f c h w -> c f h w')
        return eeg, eye, face, video, label

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert(self.head_dim*heads == embed_size) #ensure they're divisible

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
        # attention = torch.softmax(energy, dim=3)
        # print(energy.size())

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
        # self.attention = AgentAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask, inverse_mask):
        # attention, attention_matrix, attention_self, attention_other = self.attention(value, key, query, mask, inverse_mask)
        attention = self.attention(value, key, query, mask, inverse_mask)

        x = self.dropout(self.norm1(attention + query)) # Add & Norm
        forward = self.feed_forward(x) # Feed Forward
        out = self.dropout(self.norm2(forward + x)) # Add & Norm
        return out#, attention_matrix, attention_self, attention_other


class Transformer_Encoder(nn.Module):
    def __init__(self, embed_size,
                 num_layers, heads, forward_expansion,
                 dropout, max_length):
        super(Transformer_Encoder, self).__init__()
        self.embed_size = embed_size
        self.num_layers = num_layers
        # self.length_embedding = nn.Embedding(max_length, embed_size)
        # self.position_embedding = nn.Linear(max_length, embed_size)
        self.embedding = nn.Linear(max_length, embed_size)
        self.softmax = nn.Softmax(dim=2)
        # self.softmax = nn.Sigmoid()
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
        # self.final_out = nn.Linear(embed_size,3)
        # self.relu = nn.ReLU()
    
    def getPositionEncoding(self, seq_len, d, n=10000):
        P = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in np.arange(int(d / 2)):
                denominator = np.power(n, 2 * i / d)
                P[k, 2 * i] = np.sin(k / denominator)
                P[k, 2 * i + 1] = np.cos(k / denominator)
        return P

    def forward(self, x, pos_encoding, mask, inverse_mask):

        out = self.embedding(x) + pos_encoding # Need to be changed
        # for _ in range(self.num_layers):
        for layers in self.layers:
            out = layers(out, out, out, mask, inverse_mask)

        # out = self.final_out(out)
        # out = self.softmax(out)
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
        # self.final_out = nn.Linear(embed_size,3)
        # self.relu = nn.ReLU()

    def forward(self, x, pos_encoding, mask, inverse_mask):
        
        out = self.embedding(x) + pos_encoding # Need to be changed
        # for _ in range(self.num_layers):
        for layers in self.layers:
            out = layers(out,out,out,mask,inverse_mask)

        # out = self.final_out(out)
        # out = self.softmax(out)
        return out

class Aggregator_2Modal(nn.Module):
    def __init__(self, model_1, model_2, max_length, feature_embed_size, video_embed_size):
        super(Aggregator_2Modal, self).__init__()
        
        self.pos_encoding = torch.tensor(model_1.getPositionEncoding(seq_len=max_length, d=feature_embed_size, n=10000)).float().to(device)
        self.model_1 = model_1
        self.model_2 = model_2
        # self.mlp_1 = nn.Linear(int(embed_size*2), int(embed_size))
        self.final_out = nn.Linear(int(feature_embed_size * max_length + video_embed_size), 3)
        self.softmax = nn.Softmax(dim=-1)
        # self.relu = nn.ReLU()

    def forward(self, x_1, x_2, causal_mask=None, temporal_mask=None):

        out_1 = self.model_1(x_1, self.pos_encoding, causal_mask, temporal_mask)
        out_1 = torch.flatten(out_1, start_dim=1)
        out_2 = self.model_2(x_2)
        out_2 = out_2.squeeze()
        out = torch.cat((out_1, out_2), dim=1)
        # print(out.size())
        # out = self.relu(self.mlp_1(out))
        out = self.final_out(out)
        out = self.softmax(out)
        return out
    

class Trainer:
    def __init__(self, train_paths, groups, epochs, batch_size, model, optimizer, learning_rate, momentum, criterion, save_path, use_adam, modality, device, num_workers):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = model
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.num_workers = num_workers
        self.train_paths = train_paths
        self.groups = groups
        self.device = device
        self.criterion = criterion
        self.save_path = save_path
        self.use_adam = use_adam
        self.modality = modality
        self.orig_model = model
    
        
    def fit(self):

        # Using 7-fold cross validation
        group_kfold = GroupKFold(n_splits=7)

        # to store accuracies and losses        
        train_losses, train_accuracies = [], []
        val_losses, val_accuracies = [], []
        best_models = []

        for fold, (train_index, val_index) in enumerate(group_kfold.split(X=self.train_paths, groups=self.groups)):

            # initialize model
            logger.info(f"Initializing Model for fold {fold}")
            # logger.info(self.device == device)
            
            self.model = self.orig_model
            self.model = self.model.to(self.device)

            # logger.info(f"Model memory: {torch.cuda.memory_allocated(device=device)}")

            # initialize optimizer
            if self.use_adam:
                optimizer = self.optimizer(self.model.parameters(), lr=self.learning_rate)
            else:
                optimizer = self.optimizer(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
            # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)     

            # get train and validation path subsets      
            train_paths = self.train_paths[train_index]
            val_paths = self.train_paths[val_index]

            # initialize dataset
            train_dataset = createDataset(paths=train_paths)
            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)            
            val_dataset = createDataset(paths=val_paths)
            val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
            
            # set up early stopping
            early_stop_thresh = 5
            best_val_acc = -1.0
            best_train_acc = -1.0
            best_val_loss = 10000000000
            best_train_loss = 10000000000
            best_epoch = -1
            best_model = ""
            for epoch in range(1, self.epochs + 1):

                logger.info(f"Starting Training Epoch {epoch}")
                train_loss, train_acc = self.train_one_epoch(train_dataloader, optimizer)            
                logger.info(f'Training epoch {epoch}/{self.epochs} \t\t Training Loss: {train_loss / len(train_dataloader)} Training Accuracy: {train_acc}')

                val_loss, val_acc = self.validate(val_dataloader)
                logger.info(f'Validation epoch {epoch}/{self.epochs} \t\t Validation Loss: {val_loss / len(val_dataloader)} Validation Accuracy: {val_acc}')
                
                if val_loss / len(val_dataloader) < best_val_loss:
                    best_val_acc = val_acc
                    best_val_loss = val_loss / len(val_dataloader)
                    best_train_acc = train_acc
                    best_train_loss = train_loss / len(train_dataloader)
                    best_epoch = epoch
                    best_model = f"{self.save_path}/fold_{fold}_model_{epoch}.pth"
                    torch.save(self.model.state_dict(), f"{self.save_path}/fold_{fold}_model_{epoch}.pth")

                elif epoch - best_epoch > early_stop_thresh:
                    logger.info(f"Early stopped fold {fold} training at epoch {epoch}")
                    break

            val_losses.append(best_val_loss)
            val_accuracies.append(best_val_acc)
            train_losses.append(best_train_loss)
            train_accuracies.append(best_train_acc)
            best_models.append(best_model)
        
        logger.info(f"\nThe validation accuracy across folds: {val_accuracies}    average: {np.mean(val_accuracies)}")
        logger.info(f"The training accuracy across folds: {train_accuracies}    average: {np.mean(train_accuracies)}")
        logger.info(f"The validation loss across folds: {val_losses}    average: {np.mean(val_losses)}")
        logger.info(f"The training loss across folds: {train_losses}    average: {np.mean(train_losses)}")
        return best_models
            

    def train_one_epoch(self, train_dataloader, optimizer):
        self.model.train()
        train_loss = 0.0
        train_acc = 0.0
        total_correct = 0
        total_samples = 0
        for i, data in enumerate(train_dataloader):
            features = []
            eegs, eyes, faces, videos, labels = data
            if self.modality == "eeg":
                features = eegs
            elif self.modality == "eye":
                features = eyes
            elif self.modality == "face":
                features = faces

            features = features.float()
            videos = videos.float()
            features = features.to(self.device)
            
            # logger.info(f"Memory after loading features : {torch.cuda.memory_allocated(device=device)}")

            videos = videos.to(self.device)

            # logger.info(f"Memory after loading videos : {torch.cuda.memory_allocated(device=device)}")

            labels = labels.to(self.device)

            outputs = self.model(features, videos)
            predicted_labels = outputs.argmax(-1)
            loss = self.criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            # scheduler.step()
            train_loss += loss.item()
            correct = (predicted_labels == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)
        # print(total_correct, total_samples)
        train_acc = total_correct / total_samples
        return train_loss, train_acc

    def validate(self, val_dataloader):        
        self.model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_acc = 0.0
            total_correct = 0
            total_samples = 0

            for i, data in enumerate(val_dataloader):
                features = []
                eegs, eyes, faces, videos, labels = data
                if self.modality == "eeg":
                    features = eegs
                elif self.modality == "eye":
                    features = eyes
                elif self.modality == "face":
                    features = faces
                
                features = features.float()
                videos = videos.float()
                features = features.to(self.device)       
                videos = videos.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features, videos)
                predicted_labels = outputs.argmax(-1)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
                correct = (predicted_labels == labels).sum().item()
                total_correct += correct
                total_samples += labels.size(0)
        # print(total_correct, total_samples)
        val_acc = total_correct / total_samples
        return val_loss, val_acc


class Inference:
    def __init__(self, test_paths, batch_size, model, device, modality, num_workers):
        self.test_paths = test_paths
        self.batch_size = batch_size
        self.model = model
        self.device = device
        self.modality = modality
        self.num_workers = num_workers      

        test_dataset = createDataset(paths = self.test_paths)
        test_dataloader = DataLoader(test_dataset, batch_size = self.batch_size, num_workers=self.num_workers, pin_memory=True)
        self.test_dataloader = test_dataloader
    
    def test(self, model_path):
        

        logger.info(f"Loading model weights from {model_path} for inference")
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(self.device)

        self.model.eval()
        with torch.no_grad():
            test_acc = 0.0
            total_correct = 0
            total_samples = 0

            for i, data in enumerate(self.test_dataloader):
                features = []
                eegs, eyes, faces, videos, labels = data
                if self.modality == "eeg":
                    features = eegs
                elif self.modality == "eye":
                    features = eyes
                elif self.modality == "face":
                    features = faces
                
                features = features.float()
                videos = videos.float()
                features = features.to(self.device)     
                videos = videos.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features, videos)
                predicted_labels = outputs.argmax(-1)
                # print(predicted_labels)
                # print(labels)

                correct = (predicted_labels == labels).sum().item()
                total_correct += correct
                total_samples += labels.size(0)
                # print(test_acc)
        # logger.info(total_correct, total_samples)
        test_acc = total_correct / total_samples
        return test_acc

class TrainAndInference:
    def __init__(self, train_paths, test_paths, groups, epochs, batch_size, model, optimizer, learning_rate, momentum, val_step, criterion, save_path, use_adam, modality, device, num_workers):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = model
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.train_paths = train_paths
        self.test_paths = test_paths
        self.groups = groups
        self.val_step = val_step
        self.device = device
        self.criterion = criterion
        self.save_path = save_path
        self.use_adam = use_adam
        self.modality = modality
        self.num_workers = num_workers

        logger.info(f'The model will be trained and tested on {self.device}')
    
    def train_and_infer(self):
        logger.info("\nStarting Training")
        logger.info("-----------------")        
        trainer = Trainer(train_paths = self.train_paths,
                        groups = self.groups,
                        epochs = self.epochs,
                        batch_size = self.batch_size,
                        model = self.model,
                        optimizer = self.optimizer,
                        learning_rate = self.learning_rate,
                        momentum = self.momentum,
                        criterion = self.criterion,
                        save_path=self.save_path,
                        use_adam = use_adam,
                        modality = self.modality,
                        device = self.device,
                        num_workers = self.num_workers,
                    )
        best_validation_models = trainer.fit()
    
        logger.info("\nStarting Inference")
        logger.info("------------------")
        test_accuracies = []
        for fold in range(7):
            logger.info(f"Testing best validation model from fold {fold}")
            model_to_test = best_validation_models[fold]
            logger.info(model_to_test)

            tester = Inference(test_paths = self.test_paths,
                               batch_size = self.batch_size,
                               modality = self.modality,
                               model = self.model,
                               device = self.device,
                               num_workers = self.num_workers,
                            )
            
            test_acc = tester.test(model_to_test)
            test_accuracies.append(test_acc)
            logger.info(f"Test Accuracy for fold {fold}: {test_acc}")  
        
        logger.info(f"\nThe test accuracy for the best models: {test_accuracies}    average: {np.mean(test_accuracies)}")


class FinalTrainAndInference:
    def __init__(self, train_paths, test_paths, model, epochs, batch_size, optimizer, learning_rate, momentum, criterion, use_adam, save_path, modality, num_workers):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = model
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.num_workers = num_workers
        self.train_paths = train_paths
        self.test_paths = test_paths
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = criterion
        self.use_adam = use_adam
        self.save_path = save_path
        self.modality = modality
    
    def fit_predict(self):

        # initialize dataset
        train_dataset = createDataset(paths=self.train_paths)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
        
        test_dataset = createDataset(paths = self.test_paths)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

        # initialize model
        self.model = self.model.to(self.device)

        # initialize optimizer
        if self.use_adam:
            optimizer = self.optimizer(self.model.parameters(), lr=self.learning_rate)
        else:
            optimizer = self.optimizer(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)

        for epoch in range(1, self.epochs + 1):
            logger.info(f"Starting Training Epoch {epoch}")
            train_loss, train_acc = self.train_one_epoch(train_dataloader, optimizer)            
            logger.info(f'Training epoch {epoch}/{self.epochs} \t\t Training Loss: {train_loss / len(train_dataloader)} Training Accuracy: {train_acc}')

            test_loss, test_acc = self.test(test_dataloader)
            logger.info(f'Testing epoch {epoch}/{self.epochs} \t\t Testing Loss: {test_loss / len(test_dataloader)} Testing Accuracy: {test_acc}')

            torch.save(self.model.state_dict(), f"{self.save_path}/model_{epoch}.pth")

    def train_one_epoch(self, train_dataloader, optimizer):
        self.model.train()
        train_loss = 0.0
        train_acc = 0.0
        total_correct = 0
        total_samples = 0
        for i, data in enumerate(train_dataloader):
            features = []
            eegs, eyes, faces, videos, labels = data
            if self.modality == "eeg":
                features = eegs
            elif self.modality == "eye":
                features = eyes
            elif self.modality == "face":
                features = faces

            features = features.float()
            videos = videos.float()
            features = features.to(self.device)
            
            # logger.info(f"Memory after loading features : {torch.cuda.memory_allocated(device=device)}")

            videos = videos.to(self.device)

            # logger.info(f"Memory after loading videos : {torch.cuda.memory_allocated(device=device)}")

            labels = labels.to(self.device)

            outputs = self.model(features, videos)
            predicted_labels = outputs.argmax(-1)
            loss = self.criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            # scheduler.step()
            train_loss += loss.item()
            correct = (predicted_labels == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)
        # print(total_correct, total_samples)
        train_acc = total_correct / total_samples
        return train_loss, train_acc

    def test(self, test_dataloader):        
        true_labels = []
        pred_labels = []   

        self.model.eval()
        with torch.no_grad():
            test_loss = 0.0
            test_acc = 0.0
            total_correct = 0
            total_samples = 0

            for i, data in enumerate(test_dataloader):
                features = []
                eegs, eyes, faces, videos, labels = data
                if self.modality == "eeg":
                    features = eegs
                elif self.modality == "eye":
                    features = eyes
                elif self.modality == "face":
                    features = faces
                
                features = features.float()
                videos = videos.float()
                features = features.to(self.device)       
                videos = videos.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features, videos)
                predicted_labels = outputs.argmax(-1)
                loss = self.criterion(outputs, labels)

                true_labels += labels.tolist()
                pred_labels += predicted_labels.tolist()

                test_loss += loss.item()
                correct = (predicted_labels == labels).sum().item()
                total_correct += correct
                total_samples += labels.size(0)
        # print(total_correct, total_samples)
        test_acc = total_correct / total_samples
        cm = confusion_matrix(true_labels, pred_labels)
        logger.info(cm)

        return test_loss, test_acc


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    # have not yet added functionality to support multiple feature modalities (like EEG and Eye tracker)
    parser = argparse.ArgumentParser(description='Train and test multimodal models.')
    parser.add_argument("--validate", help="Use 7-fold cross validation if True, else perform inference", type=boolean_string, required=True)
    parser.add_argument("--label", help="Choose what labels to use", choices=["CL", "SA"], required=True)
    parser.add_argument("--adam", help="Whether to use Adam optimizer. If not, use SGD with momentum=0.9", type=boolean_string, required=True)
    parser.add_argument("--batchsize", help="Choose batch size for training", type=int, required=True)
    parser.add_argument("--epochs", help="Choose number of epochs for training", type=int, required=True)
    parser.add_argument("--learningrate", help="Choose learning rate for training", type=float, required=True)
    args = parser.parse_args()

    l = args.label
    v = args.validate

    modality = "face"
    image_size = 112
    feature_length = 248
    max_length = 30
    
    # Dataset paths
    train_file_paths = f"../../../multimodal_dataset_{l}_{image_size}/paths/"
    test_file_path = f"../../../multimodal_dataset_{l}_{image_size}/paths/test_paths.txt"

    groups = []
    train_paths = []
    for fold in range(7):
        with open(f"{train_file_paths}/fold_{fold}_paths.txt", "r") as f:
            fold_paths = f.readlines()
            train_paths += fold_paths
            groups += [fold] * len(fold_paths)
    
    train_paths = np.array(train_paths)
    groups = np.array(groups)
    
    with open(test_file_path, "r") as f:
        test_paths = f.readlines()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    use_adam = args.adam
    batch_size = args.batchsize
    epochs = args.epochs
    if use_adam:
        optimizer = torch.optim.Adam
    else:
        optimizer = torch.optim.SGD
    learning_rate = args.learningrate
    momentum = 0.9
    label_smoothing = 0.0
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Saving models and logs
    models_dir = ""
    logs_dir = ""
    if v == True:
        models_dir = f'../../outputs/models/validate/{l}_multimodal/dim_{image_size}_batch_{batch_size}_learning_rate_{learning_rate}_momentum_{momentum}_optimizer_{optimizer}_label_smooth_{label_smoothing}/'
        logs_dir = f'../../outputs/logs/validate/{l}_multimodal/'
    elif v == False:
        models_dir = f'../../outputs/models/test/{l}_multimodal/dim_{image_size}_batch_{batch_size}_learning_rate_{learning_rate}_momentum_{momentum}_optimizer_{optimizer}_label_smooth_{label_smoothing}/'
        logs_dir = f'../../outputs/logs/test/{l}_multimodal/'    
    
    if not os.path.isdir(models_dir):
        os.makedirs(models_dir)
    
    if not os.path.isdir(logs_dir):
        os.makedirs(logs_dir)

    filename = f"{logs_dir}/dim_{image_size}_batch_{batch_size}_learning_rate_{learning_rate}_momentum_{momentum}_optimizer_{optimizer}_label_smooth_{label_smoothing}.log"

    # video parameters
    video_embed_size = 512

    # feature parameters
    feature_embed_size = 16
    feature_num_layers = 1
    feature_head = 4
    feature_forward_expansion = 2
    feature_dropout = 0.4

    # Logger
    logging.basicConfig(filename=filename, encoding='utf-8', level=logging.DEBUG, format="%(message)s")

    logger.info("Input values")
    logger.info("---------------------")
    logger.info(f"Validate: {v}")
    logger.info(f"Model: Multimodal - MC3-18 + Facial features")
    logger.info(f"Label: {l}")    
    logger.info("---------------------")
    logger.info("Hyperparameter values")
    logger.info("---------------------") 
    logger.info(f"Feature Embed size: {feature_embed_size}")
    logger.info(f"Feature num layers: {feature_num_layers}")
    logger.info(f"Feature num head: {feature_head}")
    logger.info(f"Feature forward expansion: {feature_forward_expansion}")
    logger.info(f"Feature dropout: {feature_dropout}")
    logger.info("---------------------")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Learning Rate: {learning_rate}")
    if not use_adam:
        logger.info(f"Momentum: {momentum}")
    logger.info(f"Label Smoothing: {label_smoothing}")
    logger.info("---------------------\n")

    logger.info("Initializing video model")
    video_m = torchvision.models.video.mc3_18(weights='MC3_18_Weights.KINETICS400_V1')    

    video_model = nn.Sequential()
    for name, module in video_m.named_children():
        if name != 'fc':  # Replace 'last_layer' with the actual name of your last layer
            video_model.add_module(name, module)
    # Copy weights from original model to new model
    for name, param in video_m.named_parameters():
        if 'fc' not in name:  # Adjust if your last layer name is different
            video_model.state_dict()[name].copy_(param.data)
    
    for param in video_model.parameters():
        param.requires_grad = False

    
    logger.info("Initializing feature model")

    feature_model = Transformer_Encoder(embed_size=feature_embed_size, num_layers=feature_num_layers, heads=feature_head, \
                                        forward_expansion=feature_forward_expansion, dropout=feature_dropout, max_length=feature_length)
    feature_model.apply(he_init)
    # causal and temporal masks
    # attn_shape = (self.max_length, self.max_length)
    # causal_mask = torch.tensor(np.triu(np.ones(attn_shape), k=1).astype('uint8')).to(self.device)
    # temporal_mask = causal_mask.clone().to(self.device)
    
    logger.info("Initializing aggregated model")    
    model = Aggregator_2Modal(feature_model, video_model, max_length, feature_embed_size, video_embed_size)

    if v:        
        learner = TrainAndInference(
                        train_paths = train_paths,
                        test_paths = test_paths,
                        groups = groups,
                        epochs = epochs,
                        batch_size = batch_size,
                        model = model,
                        optimizer = optimizer,
                        learning_rate = learning_rate,
                        momentum = momentum,
                        val_step = 1,
                        criterion = criterion,
                        save_path = models_dir,
                        modality = modality,
                        use_adam = use_adam,
                        device = device,
                        num_workers = 2,
                    )                    
        learner.train_and_infer() 

    else:
        train_and_predict = FinalTrainAndInference(
                                train_paths = train_paths,
                                test_paths = test_paths,
                                model = model,
                                epochs = epochs,
                                batch_size = batch_size,
                                optimizer = optimizer,
                                learning_rate = learning_rate,
                                momentum = momentum,
                                criterion = criterion,
                                save_path = models_dir,
                                modality = modality,
                                use_adam = use_adam,
                                num_workers = 2,
                            )
        train_and_predict.fit_predict()