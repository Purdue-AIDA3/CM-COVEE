import os
import numpy as np
from scipy.io import loadmat
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
from tempfile import TemporaryFile

class DataProcesser:
    def __init__(self, data_path, user_data_path, label_directory):
        self.data_path = data_path
        self.user_data_path = user_data_path
        self.label_directory = label_directory

    def load_all_train_dataset(self):
        sub_folders = np.genfromtxt(self.data_path + 'Task 1/folds/train_fold_all.txt', dtype='str')
        all_sa_score = pd.read_csv(self.label_directory + 'situation_awareness_labels.csv')
        print(sub_folders)

        meta_psd = []
        meta_pupil_diam = []
        meta_gaze_speed = []
        meta_fixation = []
        meta_scores = []
        meta_eye_landmarks = []
        meta_face_landmarks = []
        sa_score = []

        for i in range(0, len(sub_folders)):
            id_name = sub_folders[i][1:-1]

            name_list = np.genfromtxt(self.data_path + 'NASA-TLX/' + id_name + '/Task_ID.txt', dtype='str')
            meta_scores.append(np.loadtxt(self.data_path + 'NASA-TLX/' + id_name + '/Scores.txt', delimiter=','))
            for j in range(1, len(name_list)):  # start from 1 to ignore the first video
                if 'Task_2' not in name_list[j]:
                    if not all_sa_score.name[all_sa_score.name == './' + id_name + '/' + name_list[j] + '.json'].empty:
                        pos = all_sa_score.name[all_sa_score.name=='./' + id_name + '/' + name_list[j] + '.json'].index[0]
                        all_sa_score.name[all_sa_score.name == './' + id_name + '/' + name_list[j] + '.json'].index[0]
                        sa_score.append(all_sa_score.labels[pos])
                        if 'secondary' not in name_list[j]:
                            psd = loadmat(
                                self.user_data_path + id_name + '/EEG/min_processed/log_ps/Task 1 Without Secondary Task/' +
                                name_list[
                                    j] + '.mat')
                        else:
                            psd = loadmat(
                                self.user_data_path + id_name + '/EEG/min_processed/log_ps/Task 1 With Secondary Task/' +
                                name_list[j][
                                0:-10] + '.mat')

                        meta_fixation.append(np.loadtxt(
                            self.user_data_path + id_name + '/Eye Tracker/Task 1/fixation/' + name_list[j] + '.txt'))
                        meta_pupil_diam.append(np.loadtxt(
                            self.user_data_path + id_name + '/Eye Tracker/Task 1/pupil_diam/' + name_list[j] + '.txt'))
                        meta_gaze_speed.append(np.loadtxt(
                            self.user_data_path + id_name + '/Eye Tracker/Task 1/gaze_speed/' + name_list[j] + '.txt'))
                        meta_eye_landmarks.append(np.loadtxt(
                            self.user_data_path + id_name + '/Video/Task 1/Eye Landmarks/' + name_list[j] + '.txt',
                            delimiter=','))
                        meta_face_landmarks.append(np.loadtxt(
                            self.user_data_path + id_name + '/Video/Task 1/Face Landmarks/' + name_list[j] + '.txt',
                            delimiter=','))

                        meta_psd.append(psd['all_bands'][:, 0:31, 0:40])

        return sa_score, meta_psd, meta_fixation, meta_pupil_diam, meta_gaze_speed, meta_eye_landmarks, meta_face_landmarks


    def load_train_dataset(self, num_fold):
        sub_folders = np.genfromtxt(self.data_path + 'Task 1/folds/train_fold_' + str(num_fold) + '.txt', dtype='str')
        all_sa_score = pd.read_csv(self.label_directory + 'situation_awareness_labels.csv')

        meta_psd = []
        meta_pupil_diam = []
        meta_gaze_speed = []
        meta_fixation = []
        meta_scores = []
        meta_eye_landmarks = []
        meta_face_landmarks = []
        sa_score = []

        for i in range(0, len(sub_folders)):
            id_name = sub_folders[i][1:-1]

            name_list = np.genfromtxt(self.data_path + 'NASA-TLX/' + id_name + '/Task_ID.txt', dtype='str')
            meta_scores.append(np.loadtxt(self.data_path + 'NASA-TLX/' + id_name + '/Scores.txt', delimiter=','))
            for j in range(1, len(name_list)): # start from 1 to ignore the first video
                if 'Task_2' not in name_list[j]:
                    if not all_sa_score.name[all_sa_score.name=='./' + id_name + '/' + name_list[j] + '.json'].empty:
                        pos = all_sa_score.name[all_sa_score.name=='./' + id_name + '/' + name_list[j] + '.json'].index[0]
                        sa_score.append(all_sa_score.labels[pos])
                        if 'secondary' not in name_list[j]:
                            psd = loadmat(
                                self.user_data_path + id_name + '/EEG/min_processed/log_ps/Task 1 Without Secondary Task/' + name_list[
                                    j] + '.mat')
                        else:
                            psd = loadmat(
                                self.user_data_path + id_name + '/EEG/min_processed/log_ps/Task 1 With Secondary Task/' + name_list[j][
                                                                                                          0:-10] + '.mat')

                        meta_fixation.append(np.loadtxt(self.user_data_path + id_name + '/Eye Tracker/Task 1/fixation/' + name_list[j] + '.txt'))
                        meta_pupil_diam.append(np.loadtxt(self.user_data_path + id_name + '/Eye Tracker/Task 1/pupil_diam/' + name_list[j] + '.txt'))
                        meta_gaze_speed.append(np.loadtxt(self.user_data_path + id_name + '/Eye Tracker/Task 1/gaze_speed/' + name_list[j] + '.txt'))
                        meta_eye_landmarks.append(np.loadtxt(
                            self.user_data_path + id_name + '/Video/Task 1/Eye Landmarks/' + name_list[j] + '.txt',delimiter=','))
                        meta_face_landmarks.append(np.loadtxt(
                            self.user_data_path + id_name + '/Video/Task 1/Face Landmarks/' + name_list[j] + '.txt',delimiter=','))

                        meta_psd.append(psd['all_bands'][:, 0:31, 0:40])

        return sa_score, meta_psd, meta_fixation, meta_pupil_diam, meta_gaze_speed, meta_eye_landmarks, meta_face_landmarks

    def load_val_dataset(self, num_fold):
        sub_folders = np.genfromtxt(self.data_path + 'Task 1/folds/val_fold_' + str(num_fold) + '.txt', dtype='str')
        all_sa_score = pd.read_csv(self.label_directory + 'situation_awareness_labels.csv')
        print(sub_folders)
        meta_psd = []
        meta_pupil_diam = []
        meta_gaze_speed = []
        meta_fixation = []
        meta_scores = []
        meta_eye_landmarks = []
        meta_face_landmarks = []
        sa_score = []

        for i in range(0, len(sub_folders)):
            id_name = sub_folders[i][1:-1]
            name_list = np.genfromtxt(self.data_path + 'NASA-TLX/' + id_name + '/Task_ID.txt', dtype='str')
            meta_scores.append(np.loadtxt(self.data_path + 'NASA-TLX/' + id_name + '/Scores.txt', delimiter=','))
            for j in range(1, len(name_list)):  # start from 1 to ignore the first video
                if 'Task_2' not in name_list[j]:
                    if not all_sa_score.name[all_sa_score.name=='./' + id_name + '/' + name_list[j] + '.json'].empty:
                        pos = all_sa_score.name[all_sa_score.name=='./' + id_name + '/' + name_list[j] + '.json'].index[0]
                        sa_score.append(all_sa_score.labels[pos])
                        if 'secondary' not in name_list[j]:
                            psd = loadmat(
                                self.user_data_path + id_name + '/EEG/min_processed/log_ps/Task 1 Without Secondary Task/' +
                                name_list[
                                    j] + '.mat')
                        else:
                            psd = loadmat(
                                self.user_data_path + id_name + '/EEG/min_processed/log_ps/Task 1 With Secondary Task/' +
                                name_list[j][
                                0:-10] + '.mat')

                        meta_fixation.append(np.loadtxt(
                            self.user_data_path + id_name + '/Eye Tracker/Task 1/fixation/' + name_list[j] + '.txt'))
                        meta_pupil_diam.append(np.loadtxt(
                            self.user_data_path + id_name + '/Eye Tracker/Task 1/pupil_diam/' + name_list[j] + '.txt'))
                        meta_gaze_speed.append(np.loadtxt(
                            self.user_data_path + id_name + '/Eye Tracker/Task 1/gaze_speed/' + name_list[j] + '.txt'))
                        meta_eye_landmarks.append(np.loadtxt(
                            self.user_data_path + id_name + '/Video/Task 1/Eye Landmarks/' + name_list[j] + '.txt',
                            delimiter=','))
                        meta_face_landmarks.append(np.loadtxt(
                            self.user_data_path + id_name + '/Video/Task 1/Face Landmarks/' + name_list[j] + '.txt',
                            delimiter=','))

                        meta_psd.append(psd['all_bands'][:, 0:31, 0:40])
        return sa_score, meta_psd, meta_fixation, meta_pupil_diam, meta_gaze_speed, meta_eye_landmarks, meta_face_landmarks

    def load_test_dataset(self):
        sub_folders = np.genfromtxt(self.data_path + 'Task 1/folds/test_fold.txt', dtype='str')
        all_sa_score = pd.read_csv(self.label_directory + 'situation_awareness_labels.csv')

        meta_psd = []
        meta_pupil_diam = []
        meta_gaze_speed = []
        meta_fixation = []
        meta_scores = []
        meta_eye_landmarks = []
        meta_face_landmarks = []
        sa_score = []

        for i in range(0, len(sub_folders)):
            id_name = sub_folders[i][1:-1]
            name_list = np.genfromtxt(self.data_path + 'NASA-TLX/' + id_name + '/Task_ID.txt', dtype='str')
            meta_scores.append(np.loadtxt(self.data_path + 'NASA-TLX/' + id_name + '/Scores.txt', delimiter=','))
            for j in range(1, len(name_list)):  # start from 1 to ignore the first video
                if 'Task_2' not in name_list[j]:
                    if not all_sa_score.name[all_sa_score.name=='./' + id_name + '/' + name_list[j] + '.json'].empty:
                        pos = all_sa_score.name[all_sa_score.name=='./' + id_name + '/' + name_list[j] + '.json'].index[0]
                        sa_score.append(all_sa_score.labels[pos])
                        if 'secondary' not in name_list[j]:
                            psd = loadmat(
                                self.user_data_path + id_name + '/EEG/min_processed/log_ps/Task 1 Without Secondary Task/' +
                                name_list[
                                    j] + '.mat')
                        else:
                            psd = loadmat(
                                self.user_data_path + id_name + '/EEG/min_processed/log_ps/Task 1 With Secondary Task/' +
                                name_list[j][
                                0:-10] + '.mat')

                        meta_fixation.append(np.loadtxt(
                            self.user_data_path + id_name + '/Eye Tracker/Task 1/fixation/' + name_list[j] + '.txt'))
                        meta_pupil_diam.append(np.loadtxt(
                            self.user_data_path + id_name + '/Eye Tracker/Task 1/pupil_diam/' + name_list[j] + '.txt'))
                        meta_gaze_speed.append(np.loadtxt(
                            self.user_data_path + id_name + '/Eye Tracker/Task 1/gaze_speed/' + name_list[j] + '.txt'))
                        meta_eye_landmarks.append(np.loadtxt(
                            self.user_data_path + id_name + '/Video/Task 1/Eye Landmarks/' + name_list[j] + '.txt',
                            delimiter=','))
                        meta_face_landmarks.append(np.loadtxt(
                            self.user_data_path + id_name + '/Video/Task 1/Face Landmarks/' + name_list[j] + '.txt',
                            delimiter=','))

                        meta_psd.append(psd['all_bands'][:, 0:31, 0:40])

        return sa_score, meta_psd, meta_fixation, meta_pupil_diam, meta_gaze_speed, meta_eye_landmarks, meta_face_landmarks

    def normalize_data(self, data, modality=None):
        if len(np.shape(data)) == 4:
            data = np.transpose(np.stack(data), (0, 2, 1, 3))
            original_shape = np.shape(data)
            data = data.reshape(np.shape(data)[0], np.shape(data)[1], np.shape(data)[2] * np.shape(data)[3])

            min_val = np.min(np.min(data, 2), 0)
            max_val = np.max(np.max(data, 2), 0)
            mean_val = np.mean(np.mean(data, 2), 0)
            std_val = np.std(np.std(data, 2), 0)

            for i in range(0, np.shape(data)[0]):
                for j in range(0, np.shape(data)[1]):
                    data[i, j] = (data[i, j] - min_val[j]) / (max_val[j] - min_val[j])
                    # data[i, j] = (data[i, j] - mean_val[j])/std_val[j] # for standardization

            data = data.reshape(original_shape)
            data = np.transpose(data, (0, 2, 1, 3))
            data = data.reshape(np.shape(data)[0], np.shape(data)[1], np.shape(data)[2] * np.shape(data)[3])

        else:
            data = np.stack(data)
            temp = data.flatten()
            min_val = np.min(temp, 0)
            max_val = np.max(temp, 0)
            mean_val = np.mean(temp, 0)
            std_val = np.std(temp, 0)

            if len(np.shape(data)) == 2:

                if modality == 'gaze_speed':
                    data[data > np.sqrt(2)] = np.sqrt(2)
                    max_val = np.sqrt(2)
                for i in range(0, np.shape(data)[0]):
                    if (max_val - min_val) == 0:
                        print("!!!!!!!!!!!!!!!!!!!!!!!")
                    data[i] = (data[i] - min_val) / (max_val - min_val)
                    # data[i] = ((data[i] - mean_val) / std_val) # for standardization
                data = np.stack(data)
                data = data.reshape(np.shape(data)[0], 10, int(np.shape(data)[1] / 10))

            elif len(np.shape(data)) == 3:

                for i in range(0, np.shape(data)[0]):
                    for j in range(0, np.shape(data)[1]):
                        data[i, j] = (data[i, j] - min_val) / (max_val - min_val)
                        # data[i, j] = (data[i, j] - mean_val) / std_val # for standardization
                data = np.stack(data)
                data = data.reshape(np.shape(data)[0], 10, int(np.shape(data)[2] * 30))

        return data

    def label_maker(self, levels):

        x = np.zeros((np.shape(levels)[0], 10))
        for i in range(0, np.shape(levels)[0]):
            for j in range(0, np.shape(x)[1]):
                x[i, j] = levels[i]

        return x

    def prepare_data(self, num_fold, modality, batchsize, train_levels, val_levels, test_levels,
                               train_meta_psd, val_meta_psd, test_meta_psd,
                               train_meta_pupil_diam,val_meta_pupil_diam,test_meta_pupil_diam,
                               train_meta_gaze_speed,val_meta_gaze_speed,test_meta_gaze_speed,
                               train_meta_eye_landmarks,val_meta_eye_landmarks,test_meta_eye_landmarks,
                               train_meta_face_landmarks,val_meta_face_landmarks,test_meta_face_landmarks):

        mps_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        shuffle_boolean = True

        if 'eeg' in modality:
            train_loader = DataLoader(TensorDataset(((torch.tensor(self.normalize_data(train_meta_psd))).float().to(mps_device)), (
                (torch.tensor(train_levels)).type(torch.LongTensor).to(mps_device))), batch_size=batchsize,
                                      shuffle=True, drop_last=False)
            val_loader = DataLoader(TensorDataset(((torch.tensor(self.normalize_data(val_meta_psd))).float().to(mps_device)),
                                                  ((torch.tensor(val_levels)).type(torch.LongTensor).to(mps_device))),
                                    batch_size=batchsize, shuffle=True, drop_last=False)
            test_loader = DataLoader(TensorDataset(((torch.tensor(self.normalize_data(test_meta_psd))).float().to(mps_device)),
                                                   ((torch.tensor(test_levels)).type(torch.LongTensor).to(mps_device))),
                                     batch_size=batchsize, shuffle=True, drop_last=False)

        if 'eye' in modality:
            all_eye_train = np.concatenate((self.normalize_data(train_meta_pupil_diam), self.normalize_data(train_meta_gaze_speed)), axis=2)
            all_eye_val = np.concatenate(
                (self.normalize_data(val_meta_pupil_diam), self.normalize_data(val_meta_gaze_speed)), axis=2)
            all_eye_test = np.concatenate(
                (self.normalize_data(test_meta_pupil_diam), self.normalize_data(test_meta_gaze_speed)), axis=2)

            train_loader = DataLoader(TensorDataset(((torch.tensor(all_eye_train)).float().to(mps_device)), (
                (torch.tensor(train_levels)).type(torch.LongTensor).to(mps_device))), batch_size=batchsize,
                                      shuffle=True, drop_last=False)
            val_loader = DataLoader(TensorDataset(((torch.tensor(all_eye_val)).float().to(mps_device)),
                                                  ((torch.tensor(val_levels)).type(torch.LongTensor).to(mps_device))),
                                    batch_size=batchsize, shuffle=True, drop_last=False)
            test_loader = DataLoader(TensorDataset(((torch.tensor(all_eye_test)).float().to(mps_device)),
                                                   ((torch.tensor(test_levels)).type(torch.LongTensor).to(mps_device))),
                                     batch_size=batchsize, shuffle=True, drop_last=False)

        if 'face' in modality:
            all_face_train = np.concatenate((self.normalize_data(train_meta_eye_landmarks), self.normalize_data(train_meta_face_landmarks)), axis=2)
            all_face_val = np.concatenate(
                (self.normalize_data(val_meta_eye_landmarks), self.normalize_data(val_meta_face_landmarks)), axis=2)
            all_face_test = np.concatenate(
                (self.normalize_data(test_meta_eye_landmarks), self.normalize_data(test_meta_face_landmarks)), axis=2)

            train_loader = DataLoader(TensorDataset(((torch.tensor(all_face_train)).float().to(mps_device)), (
                (torch.tensor(train_levels)).type(torch.LongTensor).to(mps_device))), batch_size=batchsize,
                                      shuffle=True, drop_last=False)
            val_loader = DataLoader(TensorDataset(((torch.tensor(all_face_val)).float().to(mps_device)),
                                                  ((torch.tensor(val_levels)).type(torch.LongTensor).to(mps_device))),
                                    batch_size=batchsize, shuffle=True, drop_last=False)
            test_loader = DataLoader(TensorDataset(((torch.tensor(all_face_test)).float().to(mps_device)),
                                                   ((torch.tensor(test_levels)).type(torch.LongTensor).to(mps_device))),
                                     batch_size=batchsize, shuffle=True, drop_last=False)

        if 'and' in modality:
            if modality == 'eeg_and_eye':
                train_loader = DataLoader(TensorDataset(((torch.tensor(self.normalize_data(train_meta_psd))).float().to(mps_device)), ((torch.tensor(all_eye_train)).float().to(mps_device)),
                                                        ((torch.tensor(train_levels)).type(torch.LongTensor).to(mps_device))),
                                        batch_size=batchsize, shuffle=True, drop_last=False)
                val_loader = DataLoader(TensorDataset(((torch.tensor(self.normalize_data(val_meta_psd))).float().to(mps_device)), ((torch.tensor(all_eye_val)).float().to(mps_device)),
                                                      ((torch.tensor(val_levels)).type(torch.LongTensor).to(mps_device))),
                                        batch_size=batchsize, shuffle=True, drop_last=False)
                test_loader = DataLoader(TensorDataset(((torch.tensor(self.normalize_data(test_meta_psd))).float().to(mps_device)), ((torch.tensor(all_eye_test)).float().to(mps_device)),
                                                       ((torch.tensor(test_levels)).type(torch.LongTensor).to(mps_device))),
                                        batch_size=batchsize, shuffle=True, drop_last=False)

            elif modality == 'eeg_and_face':
                train_loader = DataLoader(TensorDataset(((torch.tensor(self.normalize_data(train_meta_psd))).float().to(mps_device)),
                                                        ((torch.tensor(all_face_train)).float().to(mps_device)),
                                                        ((torch.tensor(train_levels)).type(torch.LongTensor).to(
                                                            mps_device))),
                                          batch_size=batchsize, shuffle=True, drop_last=False)
                val_loader = DataLoader(TensorDataset(((torch.tensor(self.normalize_data(val_meta_psd))).float().to(mps_device)),
                                                      ((torch.tensor(all_face_val)).float().to(mps_device)),
                                                      ((torch.tensor(val_levels)).type(torch.LongTensor).to(
                                                          mps_device))),
                                        batch_size=batchsize, shuffle=True, drop_last=False)
                test_loader = DataLoader(TensorDataset(((torch.tensor(self.normalize_data(test_meta_psd))).float().to(mps_device)),
                                                       ((torch.tensor(all_face_test)).float().to(mps_device)),
                                                       ((torch.tensor(test_levels)).type(torch.LongTensor).to(
                                                           mps_device))),
                                         batch_size=batchsize, shuffle=True, drop_last=False)

            elif modality == 'eye_and_face':
                train_loader = DataLoader(TensorDataset(((torch.tensor(all_eye_train)).float().to(mps_device)),
                                                        ((torch.tensor(all_face_train)).float().to(mps_device)),
                                                        ((torch.tensor(train_levels)).type(torch.LongTensor).to(
                                                            mps_device))),
                                          batch_size=batchsize, shuffle=True, drop_last=False)
                val_loader = DataLoader(TensorDataset(((torch.tensor(all_eye_val)).float().to(mps_device)),
                                                      ((torch.tensor(all_face_val)).float().to(mps_device)),
                                                      ((torch.tensor(val_levels)).type(torch.LongTensor).to(
                                                          mps_device))),
                                        batch_size=batchsize, shuffle=True, drop_last=False)
                test_loader = DataLoader(TensorDataset(((torch.tensor(all_eye_test)).float().to(mps_device)),
                                                       ((torch.tensor(all_face_test)).float().to(mps_device)),
                                                       ((torch.tensor(test_levels)).type(torch.LongTensor).to(
                                                           mps_device))),
                                         batch_size=batchsize, shuffle=True, drop_last=False)
            elif modality == 'eeg_and_eye_and_face':
                print('hello')
                train_loader = DataLoader(TensorDataset(((torch.tensor(self.normalize_data(train_meta_psd))).float().to(mps_device)),
                                                        ((torch.tensor(all_eye_train)).float().to(mps_device)),
                                                        ((torch.tensor(all_face_train)).float().to(mps_device)),
                                                        ((torch.tensor(train_levels)).type(torch.LongTensor).to(
                                                            mps_device))),
                                          batch_size=batchsize, shuffle=True, drop_last=False)
                val_loader = DataLoader(TensorDataset(((torch.tensor(self.normalize_data(val_meta_psd))).float().to(mps_device)),
                                                      ((torch.tensor(all_eye_val)).float().to(mps_device)),
                                                      ((torch.tensor(all_face_val)).float().to(mps_device)),
                                                      ((torch.tensor(val_levels)).type(torch.LongTensor).to(
                                                          mps_device))),
                                        batch_size=batchsize, shuffle=True, drop_last=False)
                test_loader = DataLoader(TensorDataset(((torch.tensor(self.normalize_data(test_meta_psd))).float().to(mps_device)),
                                                       ((torch.tensor(all_eye_test)).float().to(mps_device)),
                                                       ((torch.tensor(all_face_test)).float().to(mps_device)),
                                                       ((torch.tensor(test_levels)).type(torch.LongTensor).to(
                                                           mps_device))),
                                         batch_size=batchsize, shuffle=True, drop_last=False)

        return train_loader, val_loader, test_loader

    def prepare_npz(self, test_levels,test_meta_psd,test_meta_pupil_diam,
                    test_meta_gaze_speed,test_meta_eye_landmarks,test_meta_face_landmarks):
        test_meta_psd = self.normalize_data(test_meta_psd)
        all_eye_test = np.concatenate(
            (self.normalize_data(test_meta_pupil_diam), self.normalize_data(test_meta_gaze_speed)), axis=2)
        all_face_test = np.concatenate(
            (self.normalize_data(test_meta_eye_landmarks), self.normalize_data(test_meta_face_landmarks)), axis=2)
        np.savez_compressed('covee_src/test.npz', labels=test_levels, eeg=test_meta_psd, eye=all_eye_test, face=all_face_test)
        return











