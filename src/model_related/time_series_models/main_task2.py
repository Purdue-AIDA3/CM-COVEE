import data_loader_task2
import numpy as np
import torch
import random
import training_loop
import training_loop_multi_task2
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

# print(os.getcwd())
all_batchsize = [16, 32, 64]
all_epoch = [100, 500, 1000]
all_embed_size = [128, 256, 512]
all_layers = 3
num_layers = all_layers
head = 4
forward_expansion = 2
# drop_out = 0.4
drop_out = 0.2
all_lr = [1e-4, 5e-4, 1e-3]
modality = 'eeg_and_eye_and_face' # (eeg,eye), (eeg,face), (eye,face) the order has to be correct !!!
model_name = 'TransformerEncoder'
ground_truth = 'CL'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

directory = '/run/user/1000/gvfs/smb-share:server=datadepot.rcac.purdue.edu,share=depot/sbrunswi/data/COVEE/user_data'
user_directory = '/run/user/1000/gvfs/smb-share:server=datadepot.rcac.purdue.edu,share=depot/sbrunswi/data/COVEE/window_10_shift_10'
label_directory = 'COVEE/window_10_shift_10/'

# Initialize data processor
meta_data = data_loader_task2.DataProcesser(directory, user_directory, label_directory)

if not os.path.isdir('tmp/' + ground_truth + '/' + model_name):
    os.makedirs('tmp/' + ground_truth + '/' + model_name)

# Define cache path for the loaded data
cache_path = f'tmp/all_data_cache.pkl'
# LOAD ALL DATA ONCE AT THE BEGINNING
print("=" * 80)
print("LOADING ALL DATA")
print("=" * 80)

# Try to load from cache first, otherwise load from scratch and cache
all_data, all_labels = meta_data.load_or_cache_data(cache_path)
print(np.shape(all_labels))
# all_labels = meta_data.label_maker(all_scores)

print("\n" + "=" * 80)
print("DATA READY")
print("=" * 80)
print(f"Total samples: {len(all_labels)}")

# Split data into train/val/test (70:15:15)
from sklearn.model_selection import train_test_split

# First split: 70% train, 30% temp (val+test)
train_idx, temp_idx = train_test_split(
    np.arange(len(all_labels)), 
    test_size=0.30, 
    random_state=42
)

# Second split: split the 30% into 15% val and 15% test (50:50 of the temp set)
val_idx, test_idx = train_test_split(
    temp_idx, 
    test_size=0.50, 
    random_state=42
)

print(f"Train samples: {len(train_idx)} ({len(train_idx)/len(all_labels)*100:.1f}%)")
print(f"Val samples: {len(val_idx)} ({len(val_idx)/len(all_labels)*100:.1f}%)")
print(f"Test samples: {len(test_idx)} ({len(test_idx)/len(all_labels)*100:.1f}%)")
print()

# Extract train/val/test data
train_data = {key: val[train_idx] if val is not None else None for key, val in all_data.items()}
val_data = {key: val[val_idx] if val is not None else None for key, val in all_data.items()}
test_data = {key: val[test_idx] if val is not None else None for key, val in all_data.items()}

del all_data
import gc
gc.collect()

train_levels = all_labels[train_idx]
val_levels = all_labels[val_idx]
test_levels = all_labels[test_idx]

# Prepare data tuples for compatibility with existing code
train_meta_psd = train_data['psd']
train_meta_pupil_diam = train_data['pupil_diam']
train_meta_gaze_speed = train_data['gaze_speed']
train_meta_eye_landmarks = train_data['eye_landmarks']
train_meta_face_landmarks = train_data['face_landmarks']

val_meta_psd = val_data['psd']
val_meta_pupil_diam = val_data['pupil_diam']
val_meta_gaze_speed = val_data['gaze_speed']
val_meta_eye_landmarks = val_data['eye_landmarks']
val_meta_face_landmarks = val_data['face_landmarks']

test_meta_psd = test_data['psd']
test_meta_pupil_diam = test_data['pupil_diam']
test_meta_gaze_speed = test_data['gaze_speed']
test_meta_eye_landmarks = test_data['eye_landmarks']
test_meta_face_landmarks = test_data['face_landmarks']

# Print label samples
print("\n" + "=" * 80)
print("LABEL SAMPLES (Continuous Scores)")
print("=" * 80)

# Print first 20 training labels
print(f"\nFirst 20 training scores:")
print(f"  {train_levels[:20]}")

for batchsize in all_batchsize:
    for epoch in all_epoch:
        for embed_size in all_embed_size:
            for lr in all_lr:
                # for num_layers in all_layers:

                print('\n')
                print('\n')
                print("====================================================================")
                print(f"Training with: batch_size={batchsize}, epoch={epoch}, embed_size={embed_size}, lr={lr}")
                print("====================================================================")
                print('\n')
                
                # Check if cached dataloaders exist
                cache_file = f"tmp/{ground_truth}/{modality}_train_70_15_15.pth"
                if os.path.exists(cache_file):
                    print('Loading cached dataloaders...')
                    train_loader = torch.load(f"tmp/{ground_truth}/{modality}_train_70_15_15.pth", weights_only=False)
                    val_loader = torch.load(f"tmp/{ground_truth}/{modality}_val_70_15_15.pth", weights_only=False)
                    test_loader = torch.load(f"tmp/{ground_truth}/{modality}_test_70_15_15.pth", weights_only=False)
                    train_loader = change_batch_size(train_loader, batchsize)
                    val_loader = change_batch_size(val_loader, batchsize)
                    test_loader = change_batch_size(test_loader, batchsize)
                else:
                    print('Creating dataloaders from split data...')
                    train_loader, val_loader, test_loader = meta_data.prepare_data(
                        fold_num=1,  # Not used anymore but kept for compatibility
                        modality=modality, 
                        batch_size=batchsize,
                        train_levels=train_levels, 
                        val_levels=val_levels, 
                        test_levels=test_levels,
                        train_psd=train_meta_psd, 
                        val_psd=val_meta_psd, 
                        test_psd=test_meta_psd,
                        train_pupil=train_meta_pupil_diam,
                        val_pupil=val_meta_pupil_diam,
                        test_pupil=test_meta_pupil_diam,
                        train_gaze=train_meta_gaze_speed,
                        val_gaze=val_meta_gaze_speed,
                        test_gaze=test_meta_gaze_speed,
                        train_eye_landmarks=train_meta_eye_landmarks,
                        val_eye_landmarks=val_meta_eye_landmarks,
                        test_eye_landmarks=test_meta_eye_landmarks,
                        train_face_landmarks=train_meta_face_landmarks,
                        val_face_landmarks=val_meta_face_landmarks,
                        test_face_landmarks=test_meta_face_landmarks
                    )
                    
                    # Cache the dataloaders
                    if not os.path.isdir(f'tmp/{ground_truth}'):
                        os.makedirs(f'tmp/{ground_truth}')
                    torch.save(train_loader, f"tmp/{ground_truth}/{modality}_train_70_15_15.pth")
                    torch.save(val_loader, f"tmp/{ground_truth}/{modality}_val_70_15_15.pth")
                    torch.save(test_loader, f"tmp/{ground_truth}/{modality}_test_70_15_15.pth")
                    print('Dataloaders cached for future use.')

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

                        TrainingLoop = training_loop_multi_task2.TrainingLoop(modality, train_loader, val_loader, batchsize, epoch, embed_size,
                                                                  num_layers, head, forward_expansion, drop_out, max_length, [feature_length_1,
                                                                  feature_length_2], device, lr, model_name, ground_truth, Task_2=True)
                        model, all_train_mse, all_val_mse, train_acc, val_acc = TrainingLoop.training_multi()
                    else:
                        feature_length_1 = int(31 * 40)
                        feature_length_2 = 60
                        feature_length_3 = 7440
                        TrainingLoop = training_loop_multi_task2.TrainingLoop(modality, train_loader, val_loader,
                                                                        batchsize, epoch, embed_size,
                                                                        num_layers, head, forward_expansion,
                                                                        drop_out, max_length, [feature_length_1,
                                                                        feature_length_2, feature_length_3], device, lr, model_name,
                                                                        ground_truth, Task_2=True)
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

                data = {
                    # 'Training Accuracy': train_acc[-1],
                    # 'Validation Accuracy': val_acc[-1],
                    'Batch Size': batchsize,
                    'Epoch': epoch,
                    'Number of Layers': num_layers,
                    'Embed Size': embed_size,
                    'Learning Rate': lr,
                    'Head': head,
                    'Forward Expansion': forward_expansion,
                    'Dropout': drop_out,
                    'Training MSE': all_train_mse[-1],
                    'Validation MSE': all_val_mse[-1]
                }
                new_df = pd.DataFrame(data, index=[0])
                if os.path.isfile(
                    'tmp/' + ground_truth + '/' + model_name + '/' + modality + '_outputs.csv'):
                    new_df.to_csv('tmp/' + ground_truth + '/' + model_name + '/' + modality + '_outputs.csv', mode='a', header=False, index=False)
                else:
                    new_df.to_csv('tmp/' + ground_truth + '/' + model_name + '/' + modality + '_outputs.csv', index=False)