import data_loader, data_loader_SA
import numpy as np
import torch
import random
import test_loop
from Transformer_Encoder import Transformer_Encoder
from LSTM import LSTM
from GRU import GRU

torch.manual_seed(8000)
random.seed(8000)
np.random.seed(8000)

embed_size = 64
modality = 'face'
model_name = 'GRU' # LSTM or TransformerEncoder or GRU
ground_truth = 'SA'

directory = '/Users/sky/Downloads/Subject Data/'
if ground_truth == 'CL':
    meta_data = data_loader.DataProcesser(directory)
elif ground_truth == 'SA':
    meta_data = data_loader_SA.DataProcesser(directory)

test_meta_scores, test_meta_psd, test_meta_fixation, test_meta_pupil_diam, test_meta_gaze_speed, test_meta_eye_landmarks, test_meta_face_landmarks = meta_data.load_test_dataset()
test_levels = meta_data.label_maker(test_meta_scores)

test_loader = meta_data.prepare_test_data(modality, 64,test_levels,test_meta_psd,test_meta_pupil_diam,test_meta_gaze_speed,test_meta_eye_landmarks,test_meta_face_landmarks)


max_length = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if modality == 'eeg':
    feature_length = int(31 * 40)
elif modality == 'eye':
    feature_length = 60
elif modality == 'face':
    feature_length = 7440

if model_name == 'TransformerEncoder':
    model = Transformer_Encoder(embed_size=embed_size, num_layers=1, heads=4, \
                                forward_expansion=2, dropout=0.4,
                                max_length=feature_length).to(device)
elif model_name == 'LSTM':
    model = LSTM(embed_size=embed_size, feature_size=feature_length)
elif model_name == 'GRU':
    model = GRU(embed_size=embed_size, feature_size=feature_length)

model.load_state_dict(torch.load('../' + model_name + '/model_' + modality + '.pt'))

test_loop.testing(model, model_name, embed_size, test_loader, device)








