## general imports
import random
import math
import os
import logging
from os.path import join as pjoin
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

## torchvision
import torchvision

## einops
from einops import rearrange, repeat, reduce

## scipy
# import scipy

## scikit-learn
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix

## hugging_face 
from transformers import VivitForVideoClassification, VivitConfig

## set seeds
random.seed(8000)
torch.manual_seed(8000)
np.random.seed(8000)
rng = np.random.RandomState(seed=8000)

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

class createDataset(Dataset):
    def __init__(self, paths, model_type):
        self.paths = paths
        self.model_type = model_type

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        data = np.load(self.paths[index][:-1])
        input = data["video"]
        label = data["label"]
        if self.model_type != "vivit":
            input = rearrange(input, 'f c h w -> c f h w')
        return input, label


class Trainer:
    def __init__(self, train_paths, groups, epochs, batch_size, model, model_type, optimizer, learning_rate, momentum, criterion, save_path, use_adam, pretrained, device, num_workers):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = model
        self.model_type = model_type
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
        self.pretrained = pretrained
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
            logger.info(f"Initializing model for fold {fold}")

            self.model = self.orig_model            
            self.model = self.model.to(self.device)

            # initialize optimizer
            if self.use_adam:
                optimizer = self.optimizer(self.model.parameters(), lr=self.learning_rate)
            else:
                optimizer = self.optimizer(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=0.001)
            # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)     

            # get train and validation path subsets      
            train_paths = self.train_paths[train_index]
            val_paths = self.train_paths[val_index]

            # initialize dataset
            train_dataset = createDataset(paths=train_paths, model_type=self.model_type)
            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)            
            val_dataset = createDataset(paths=val_paths, model_type=self.model_type)
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

                logger.info("Starting Training Epoch " + str(epoch))
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
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(inputs)
            if self.model_type == "vivit":
                outputs = outputs.logits            
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
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                if self.model_type == "vivit":
                    outputs = outputs.logits
                predicted_labels = outputs.argmax(-1)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
                correct = (predicted_labels == labels).sum().item()
                total_correct += correct
                total_samples += labels.size(0)

        val_acc = total_correct / total_samples
        return val_loss, val_acc


class Inference:
    def __init__(self, test_paths, batch_size, device, model, model_type, num_workers):
        self.test_paths = test_paths
        self.batch_size = batch_size
        self.device = device
        self.model = model
        self.model_type = model_type
        self.num_workers = num_workers                

        test_dataset = createDataset(paths = self.test_paths, model_type=self.model_type)
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
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = model(inputs)
                if self.model_type == "vivit":
                    outputs = outputs.logits
                predicted_labels = outputs.argmax(-1)
                # logger.info(predicted_labels)
                # logger.info(labels)

                correct = (predicted_labels == labels).sum().item()
                total_correct += correct
                total_samples += labels.size(0)
                # logger.info(test_acc)
        # logger.info(total_correct, total_samples)
        test_acc = total_correct / total_samples
        return test_acc

class TrainAndValidate:
    def __init__(self, train_paths, test_paths, groups, epochs, batch_size, model, model_type, optimizer, learning_rate, momentum, criterion, save_path, use_adam, device, pretrained, num_workers):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = model
        self.model_type = model_type
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.train_paths = train_paths
        self.test_paths = test_paths
        self.groups = groups
        self.device = device
        self.criterion = criterion
        self.save_path = save_path
        self.use_adam = use_adam
        self.pretrained = pretrained
        self.num_workers = num_workers

        logger.info(f'The model will be trained and tested on {self.device}')
    
    def train_and_validate(self):
        logger.info("\nStarting Training")
        logger.info("-----------------")        
        trainer = Trainer(train_paths = self.train_paths,
                        groups = self.groups,
                        epochs = self.epochs,
                        batch_size = self.batch_size,
                        model = self.model,
                        model_type = self.model_type,
                        optimizer = self.optimizer,
                        learning_rate = self.learning_rate,
                        momentum = self.momentum,
                        criterion=self.criterion,
                        save_path=self.save_path,
                        use_adam = use_adam,
                        pretrained = self.pretrained,
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
                               model = self.model,
                               model_type = self.model_type,
                               device = self.device,
                               num_workers = self.num_workers,
                            )
            
            test_acc = tester.test(model_to_test)
            test_accuracies.append(test_acc)
            logger.info(f"Test Accuracy for fold {fold}: {test_acc}")  
        
        logger.info(f"\nThe test accuracy for the best models: {test_accuracies}    average: {np.mean(test_accuracies)}")


class TrainAndPredict:
    def __init__(self, train_paths, test_paths, epochs, batch_size, model, model_type, optimizer, learning_rate, momentum, criterion, save_path, use_adam, device, pretrained, num_workers):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = model
        self.model_type = model_type
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.num_workers = num_workers
        self.train_paths = train_paths
        self.test_paths = test_paths
        self.device = device
        self.criterion = criterion
        self.save_path = save_path
        self.use_adam = use_adam
        self.pretrained = pretrained
    
    def fit_predict(self):

        # initialize dataset
        train_dataset = createDataset(paths=self.train_paths, model_type=self.model_type)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
        
        test_dataset = createDataset(paths = self.test_paths, model_type=self.model_type)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
       
        self.model = self.model.to(self.device)

        # initialize optimizer
        if self.use_adam:
            optimizer = self.optimizer(self.model.parameters(), lr=self.learning_rate)
        else:
            optimizer = self.optimizer(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=0.001)

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
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(inputs)
            if self.model_type == "vivit":
                outputs = outputs.logits
            predicted_labels = outputs.argmax(-1)
            loss = self.criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            # logger.info(f"Learning rate for iteration: {optimizer.param_groups[0]["lr"]}")
            train_loss += loss.item()
            correct = (predicted_labels == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)

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
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                
                if self.model_type == "vivit":
                    outputs = outputs.logits
                predicted_labels = outputs.argmax(-1)
                loss = self.criterion(outputs, labels)
                
                true_labels += labels.tolist()
                pred_labels += predicted_labels.tolist()

                test_loss += loss.item()
                correct = (predicted_labels == labels).sum().item()
                total_correct += correct
                total_samples += labels.size(0)

        test_acc = total_correct / total_samples
        cm = confusion_matrix(true_labels, pred_labels)
        logger.info(cm)

        return test_loss, test_acc


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description='Train and test video vision models.')
    parser.add_argument("--validate", help="Use 7-fold cross validation if True, else perform inference", type=boolean_string, required=True)
    parser.add_argument("--model", help="Choose model to use", choices=["resnet3d", "mixedconv", "r2p1d", "vivit"], required=True)
    parser.add_argument("--label", help="Choose what labels to use", choices=["CL", "SA"], required=True)
    parser.add_argument("--adam", help="Whether to use Adam optimizer. If not, use SGD with momentum=0.9", type=boolean_string, required=True)
    parser.add_argument("--batchsize", help="Choose batch size for training", type=int, required=True)
    parser.add_argument("--epochs", help="Choose number of epochs for training", type=int, required=True)
    parser.add_argument("--learningrate", help="Choose learning rate for training", type=float, required=True)
    parser.add_argument("--pretrained", help="Whether to load Kinetics 400 weights", type=boolean_string, required=True)
    args = parser.parse_args()
    
    # # train model and use 7-fold cross validation for model generalizability, also test best models from all folds to get a final average test accuracy
    # # this average test accuracy will be used later to compare different sets of hyperparameters and choose the best set

    m = args.model
    l = args.label
    v = args.validate

    image_size = 224 if m == "vivit" else 112

    # Dataset paths
    train_file_paths = f"../../../video_dataset_{l}_{image_size}_32frames/paths/"
    test_file_path = f"../../../video_dataset_{l}_{image_size}_32frames/paths/test_paths.txt"

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
    pretrained = args.pretrained
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Saving models and logs
    models_dir = ""
    logs_dir = ""
    if v == True:
        models_dir = f'../../outputs/models/validate/{l}_{m}/dim_{image_size}_batch_{batch_size}_learning_rate_{learning_rate}_momentum_{momentum}_optimizer_{optimizer}_label_smooth_{label_smoothing}_pretrained_{pretrained}/'
        logs_dir = f'../../outputs/logs/validate/{l}_{m}/'
    elif v == False:
        models_dir = f'../../outputs/models/test/{l}_{m}/dim_{image_size}_batch_{batch_size}_learning_rate_{learning_rate}_momentum_{momentum}_optimizer_{optimizer}_label_smooth_{label_smoothing}_pretrained_{pretrained}/'
        logs_dir = f'../../outputs/logs/test/{l}_{m}/'    
    
    if not os.path.isdir(models_dir):
        os.makedirs(models_dir)
    
    if not os.path.isdir(logs_dir):
        os.makedirs(logs_dir)

    filename = f"{logs_dir}/dim_{image_size}_batch_{batch_size}_learning_rate_{learning_rate}_momentum_{momentum}_optimizer_{optimizer}_label_smooth_{label_smoothing}_pretrained_{pretrained}.log"

    # Logger
    logging.basicConfig(filename=filename, encoding='utf-8', level=logging.DEBUG, format="%(message)s")

    logger.info("Input values")
    logger.info("---------------------")
    logger.info(f"Validate: {v}")
    logger.info(f"Model: {m}")
    logger.info(f"Label: {l}")    
    logger.info("---------------------")
    logger.info("Hyperparameter values")
    logger.info("---------------------")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Learning Rate: {learning_rate}")
    if not use_adam:
        logger.info(f"Momentum: {momentum}")
    logger.info(f"Label Smoothing: {label_smoothing}")
    logger.info(f"Pretrained: {pretrained}")
    logger.info("---------------------\n")

    # Choose model
    if m == "vivit":
        configuration = VivitConfig()
        if pretrained:
            model = VivitForVideoClassification(configuration).from_pretrained("google/vivit-b-16x2-kinetics400")
        else:
            model = VivitForVideoClassification(configuration)
        model.classifier = nn.Linear(in_features=model.config.hidden_size, out_features=3)

        for param in model.parameters():
            param.requires_grad = False        
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        if m == "resnet3d":
            if pretrained:
                model = torchvision.models.video.r3d_18(weights='R3D_18_Weights.KINETICS400_V1')
            else:
                model = torchvision.models.video.r3d_18()
        elif m == "mixedconv":   
            if pretrained:
                model = torchvision.models.video.mc3_18(weights='MC3_18_Weights.KINETICS400_V1')
            else:
                model = torchvision.models.video.mc3_18()
        elif m == "r2p1d":        
            if pretrained:
                model = torchvision.models.video.r2plus1d_18(weights='R2Plus1D_18_Weights.KINETICS400_V1')
            else:
                model = torchvision.models.video.r2plus1d_18()  
        else:
            logger.info("Model not implemented, exiting")
            exit(-1)        
        model.fc = nn.Linear(model.fc.in_features, 3)

        for param in model.parameters():
            param.requires_grad = False        
        for param in model.fc.parameters():
            param.requires_grad = True
    
    # for name, param in model.named_parameters():
    #     print(name, param)

    logger.info("---------------------\n")
    logger.info("Model Architecture")
    logger.info(f"{model}\n")
    logger.info("---------------------\n")

    if args.validate:
        logger.info("Starting Training and Cross Validation\n")
        learner = TrainAndValidate(train_paths = train_paths,
                                    test_paths = test_paths,
                                    groups = groups,
                                    epochs = epochs,
                                    batch_size = batch_size,
                                    model = model,
                                    model_type = m,
                                    optimizer = optimizer,
                                    learning_rate = learning_rate,
                                    momentum = momentum,
                                    criterion=nn.CrossEntropyLoss(label_smoothing=label_smoothing),
                                    save_path = models_dir,
                                    use_adam = use_adam,
                                    device = device,
                                    pretrained = pretrained,
                                    num_workers = 2,
                                )                    
        learner.train_and_validate()
    else:
        logger.info("Starting Training and Inference\n")
        train_and_predict = TrainAndPredict(
                                train_paths = train_paths,
                                test_paths = test_paths,
                                epochs = epochs,
                                batch_size = batch_size,
                                model = model,
                                model_type = m,
                                optimizer = optimizer,
                                learning_rate = learning_rate,
                                momentum = momentum,
                                criterion = criterion,
                                save_path = models_dir,
                                use_adam = use_adam,
                                device = device,
                                pretrained = pretrained,
                                num_workers = 2,
                            )
        train_and_predict.fit_predict()