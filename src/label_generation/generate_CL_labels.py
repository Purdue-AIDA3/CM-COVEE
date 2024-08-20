import os
import numpy as np
import pandas as pd

directory = '/COVEE/simple_datasets/NASA-TLX/'
sub_folders = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
sub_folders.sort()
print(sub_folders)

for i in range(0, len(sub_folders)):
    id_name = sub_folders[i]
    name_list = np.genfromtxt(directory + id_name + '/Task_ID.txt', dtype='str')
    scores = np.loadtxt(directory + id_name + '/Scores.txt', delimiter=',')
    for j in range(1, len(name_list)):
        if 'Task_2' not in name_list[j]:
            if 1 <= scores[i,0] <= 7:
                levels = 0
            elif 8 <= scores[i,0] <= 13:
                levels = 1
            elif 14 <= scores[i,0] <= 20:
                levels = 2

            data = {
                'Labels': levels,
                'User ID': id_name,
                'Task ID': name_list[j]
            }
            new_df = pd.DataFrame(data, index=[0])
            if os.path.isfile(
                    'test_cl_labels.csv'):
                new_df.to_csv('test_cl_labels.csv',
                              mode='a',
                              header=False, index=False)
            else:
                new_df.to_csv('test_cl_labels.csv',
                              index=False)