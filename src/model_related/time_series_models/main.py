import data_loader
import data_loader_SA
import numpy as np
import torch
import random
import training_loop
import training_loop_multi
import pandas as pd
import os
from torch.utils.data import DataLoader

torch.manual_seed(8000)
random.seed(8000)
np.random.seed(8000)

def change_batch_size(loader, new_batch_size):
    new_train_loader = DataLoader(
        dataset=loader.dataset,
        batch_size=new_batch_size,
        shuffle=True,
        drop_last=False
    )
    return new_train_loader

all_batchsize = [32, 64, 128]
all_epoch = [100, 500, 1000]
all_embed_size = [16, 32, 64, 128, 256, 512]
all_layers = 1
num_layers = all_layers
head = 4
forward_expansion = 2
# drop_out = 0.4
drop_out = 0.1
all_lr = [1e-6, 1e-5, 1e-4]
modality = 'eeg_and_eye_and_face' # (eeg,eye), (eeg,face), (eye,face) the order has to be correct !!!
model_name = 'TransformerEncoder'
ground_truth = 'SA'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

directory = 'COVEE/simple_datasets/'
user_directory = 'COVEE/user_data/'
label_directory = 'COVEE/src/label_generation/'

if ground_truth == 'CL':
    meta_data = data_loader.DataProcesser(directory, user_directory, label_directory)
elif ground_truth == 'SA':
    meta_data = data_loader_SA.DataProcesser(directory, user_directory, label_directory)

if not os.path.isdir('covee_src/' + ground_truth + '/' + model_name):
    os.makedirs('covee_src/' + ground_truth + '/' + model_name)

test_meta_scores, test_meta_psd, test_meta_fixation, test_meta_pupil_diam, test_meta_gaze_speed, test_meta_eye_landmarks, test_meta_face_landmarks = meta_data.load_test_dataset()
test_levels = meta_data.label_maker(test_meta_scores)

for batchsize in all_batchsize:
    for epoch in all_epoch:
        for embed_size in all_embed_size:
            for lr in all_lr:
                # for num_layers in all_layers:

                all_train_loss = []
                all_train_acc = []
                all_val_loss = []
                all_val_acc = []
                for num_fold in range(1, 8):
                    print('\n')
                    print('\n')
                    print("====================================================================")
                    print(num_fold)
                    print("====================================================================")
                    print('\n')
                    print('\n')
                    if os.path.exists(f"covee_src/data/{ground_truth}/{modality}_train_{num_fold}.pth"):
                        print('hello')
                        train_loader = torch.load(f"covee_src/data/{ground_truth}/{modality}_train_{num_fold}.pth", weights_only=False)
                        val_loader = torch.load(
                            f"covee_src/data/{ground_truth}/{modality}_val_{num_fold}.pth", weights_only=False)
                        test_loader = torch.load(
                            f"covee_src/data/{ground_truth}/{modality}_test_{num_fold}.pth", weights_only=False)
                        train_loader = change_batch_size(train_loader, batchsize)
                        val_loader = change_batch_size(val_loader, batchsize)
                        test_loader = change_batch_size(test_loader, batchsize)
                    else:
                        train_meta_scores, train_meta_psd, train_meta_fixation, train_meta_pupil_diam, train_meta_gaze_speed, train_meta_eye_landmarks, train_meta_face_landmarks = meta_data.load_train_dataset(num_fold)
                        val_meta_scores, val_meta_psd, val_meta_fixation, val_meta_pupil_diam, val_meta_gaze_speed, val_meta_eye_landmarks, val_meta_face_landmarks = meta_data.load_val_dataset(num_fold)
                        train_levels = meta_data.label_maker(train_meta_scores)
                        val_levels = meta_data.label_maker(val_meta_scores)



                        train_loader, val_loader, test_loader = meta_data.prepare_data(num_fold, modality, batchsize,
                                                                                       train_levels, val_levels, test_levels,
                                                                                       train_meta_psd, val_meta_psd, test_meta_psd,
                                                                                       train_meta_pupil_diam,val_meta_pupil_diam,test_meta_pupil_diam,
                                                                                       train_meta_gaze_speed,val_meta_gaze_speed,test_meta_gaze_speed,
                                                                                       train_meta_eye_landmarks,val_meta_eye_landmarks,test_meta_eye_landmarks,
                                                                                       train_meta_face_landmarks,val_meta_face_landmarks,test_meta_face_landmarks)
                        torch.save(train_loader,
                            f"covee_src/data/{ground_truth}/{modality}_train_{num_fold}.pth")
                        torch.save(val_loader,
                            f"covee_src/data/{ground_truth}/{modality}_val_{num_fold}.pth")
                        torch.save(test_loader,
                            f"covee_src/data/{ground_truth}/{modality}_test_{num_fold}.pth")

                    max_length = 10
                    if 'and' in modality:
                        if modality.count('and') == 1:
                            if 'eeg' in modality and 'eye' in modality:
                                feature_length_1 = int(31*40)
                                feature_length_2 = 60
                            elif 'eeg' in modality and 'face' in modality:
                                feature_length_1 = int(31*40)
                                feature_length_2 = 7440
                            elif 'eye' in modality and 'face' in modality:
                                feature_length_1 = 60
                                feature_length_2 = 7440

                            TrainingLoop = training_loop_multi.TrainingLoop(modality, train_loader, val_loader, batchsize, epoch, embed_size,
                                                                      num_layers, head, forward_expansion, drop_out, max_length, [feature_length_1,
                                                                      feature_length_2], device, lr, model_name, ground_truth)
                            model, all_train_mse, all_val_mse, train_acc, val_acc = TrainingLoop.training_multi()
                        else:
                            feature_length_1 = int(31 * 40)
                            feature_length_2 = 60
                            feature_length_3 = 7440
                            TrainingLoop = training_loop_multi.TrainingLoop(modality, train_loader, val_loader,
                                                                            batchsize, epoch, embed_size,
                                                                            num_layers, head, forward_expansion,
                                                                            drop_out, max_length, [feature_length_1,
                                                                            feature_length_2, feature_length_3], device, lr, model_name,
                                                                            ground_truth)
                            model, all_train_mse, all_val_mse, train_acc, val_acc = TrainingLoop.training_multi()

                    else:
                        if modality == 'eeg':
                            feature_length = int(31*40)
                        elif modality == 'eye':
                            feature_length = 60
                        elif modality == 'face':
                            feature_length = 7440

                        TrainingLoop = training_loop.TrainingLoop(modality, train_loader, val_loader, batchsize, epoch, embed_size, num_layers, head, forward_expansion, drop_out, max_length, feature_length, device, lr, model_name, ground_truth)
                        model, all_train_mse, all_val_mse, train_acc, val_acc = TrainingLoop.training()

                    all_train_loss.append(all_train_mse)
                    all_val_loss.append(all_val_mse)
                    all_train_acc.append(train_acc)
                    all_val_acc.append(val_acc)


                all_train_loss = np.mean(np.stack(all_train_loss),axis=0)
                all_val_loss = np.mean(np.stack(all_val_loss),axis=0)
                all_train_acc = np.mean(np.stack(all_train_acc),axis=0)
                all_val_acc = np.mean(np.stack(all_val_acc),axis=0)


                data = {
                    'Averaged Training Accuracy': all_train_acc[-1],
                    'Averaged Validation Accuracy': all_val_acc[-1],
                    'Batch Size': batchsize,
                    'Epoch': epoch,
                    'Number of Layers': num_layers,
                    'Embed Size': embed_size,
                    'Learning Rate': lr,
                    'Head': head,
                    'Forward Expansion': forward_expansion,
                    'Dropout': drop_out
                }
                new_df = pd.DataFrame(data, index=[0])
                if os.path.isfile(
                    'covee_src/' + ground_truth + '/' + model_name + '/' + modality + '_outputs.csv'):
                    new_df.to_csv('covee_src/' + ground_truth + '/' + model_name + '/' + modality + '_outputs.csv', mode='a', header=False, index=False)
                else:
                    new_df.to_csv('covee_src/' + ground_truth + '/' + model_name + '/' + modality + '_outputs.csv', index=False)






