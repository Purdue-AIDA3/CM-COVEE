import os
import pandas as pd
import time
import datetime
import json
import numpy as np


folder = '/COVEE/user_data/'

task_list = ['Task2_one_platform','Task2_one_platform_secondary','Task2_two_platforms','Task2_two_platforms_secondary']

sub_folders = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]

print(sub_folders)

for i in range(0, len(sub_folders)):
    id_name = sub_folders[i]
    for k in range(0, 4):
        recording_location = folder+id_name+'/Eye Tracker/Task 2/Trial ' + str(k+1) + '/00'+str(k+1)
        if os.path.isfile(recording_location + '/info.player.json') and \
            os.path.isfile(folder + id_name+"/clicks_data/"+task_list[k]+'.json') and \
                os.path.isfile(folder + id_name+"/window_times.json"):
            f = open(recording_location + '/info.player.json')
            y = json.load(f)
            start_time_system = y["start_time_system_s"] # unix epoch timestamp
            start_time_synced = y["start_time_synced_s"] # pupil epoch timestamp
            offset = start_time_system - start_time_synced

            exported_pupil_csv = os.path.join(recording_location+'/exports/000/pupil_positions.csv')
            pupil_pd_frame = pd.read_csv(exported_pupil_csv)
            exported_gaze_csv = os.path.join(recording_location+'/exports/000/gaze_positions.csv')
            gaze_data = pd.read_csv(exported_gaze_csv)
            # exported_fixation_csv = os.path.join(recording_location+'/exports/000/fixations.csv')
            # fixation_data = pd.read_csv(exported_fixation_csv)

            f_superbase = open(folder + id_name+"/clicks_data/"+task_list[k]+'.json')
            superbase_data = json.load(f_superbase)

            f_windows = open(folder + id_name+"/window_times.json")
            window_times = json.load(f_windows)

            video_end_timestamp = time.mktime(((datetime.datetime.strptime(window_times['Task_2'+task_list[k][5:]]['end_time'],
                                                                             "%Y-%m-%d %H:%M:%S.%f")) + datetime.timedelta(
                hours=0)).timetuple())
            video_start_timestamp = time.mktime(((datetime.datetime.strptime(superbase_data['mission_start_time'],
                                                                           "%Y-%m-%d %H:%M:%S.%f")) + datetime.timedelta(
                hours=0)).timetuple())
            video_start_timestamp = start_time_synced + video_start_timestamp - start_time_system
            video_end_timestamp = start_time_synced + video_end_timestamp - start_time_system


            idx_start = np.argmin(abs(gaze_data['gaze_timestamp'] - video_start_timestamp))
            idx_end = np.argmin(abs(gaze_data['gaze_timestamp'] - video_end_timestamp))
            if idx_start==idx_end:
                print(id_name)
                raise Exception("Bad ID!")

            world_start = np.where(gaze_data['world_index'] == gaze_data['world_index'][idx_start])[0][0]
            world_end = np.where(gaze_data['world_index'] == gaze_data['world_index'][idx_end])[0][-1]

            video_gaze = gaze_data[world_start:world_end + 1]

            idx_start = np.argmin(abs(pupil_pd_frame['pupil_timestamp'] - video_start_timestamp))
            idx_end = np.argmin(abs(pupil_pd_frame['pupil_timestamp'] - video_end_timestamp))

            idx_start = np.where(pupil_pd_frame['world_index'] == pupil_pd_frame['world_index'][idx_start])[0][0]
            idx_end = np.where(pupil_pd_frame['world_index'] == pupil_pd_frame['world_index'][idx_end])[0][-1]


            # fixation_start = next(x for x, val in enumerate(fixation_data['start_frame_index'])
            #                             if val >= pupil_pd_frame['world_index'][idx_start])
            # fixation_end = next(x for x, val in enumerate(fixation_data['start_frame_index'])
            #                             if val >= pupil_pd_frame['world_index'][idx_end])-1

            # if (pupil_pd_frame['world_index'][idx_end] - pupil_pd_frame['world_index'][idx_start])>=300:
            temp_fixation_frame = np.zeros(
                (pupil_pd_frame['world_index'][idx_end] - pupil_pd_frame['world_index'][idx_start],))
            # else:

            # for j in range(0,fixation_end-fixation_start):
            #     temp_fixation_frame[(fixation_data['start_frame_index'][fixation_start+j]-pupil_pd_frame['world_index'][idx_start]):
            #     (fixation_data['end_frame_index'][fixation_start+j]-pupil_pd_frame['world_index'][idx_start])] = 1

            temp = pupil_pd_frame[idx_start:idx_end + 1]
            video_pupil = temp[temp.method == 'pye3d 0.3.0 real-time']

            x, unique_world_idx = np.unique(video_pupil['world_index'], return_index=True)
            temp = []
            for j in range(0, -1 + np.shape(unique_world_idx)[0]):
                temp.append(np.mean(video_pupil['diameter'][unique_world_idx[j]:unique_world_idx[j + 1]]))
            temp.append(np.mean(video_pupil['diameter'][unique_world_idx[-1]:]))
            idx = 0

            y, unique_world_idx_gaze = np.unique(video_gaze['world_index'], return_index=True)
            temp_pos_x = []
            temp_pos_y = []
            for j in range(0, -1 + np.shape(unique_world_idx_gaze)[0]):
                temp_pos_x.append(
                    np.mean(video_gaze['norm_pos_x'][unique_world_idx_gaze[j]:unique_world_idx_gaze[j + 1]]))
                temp_pos_y.append(
                    np.mean(video_gaze['norm_pos_y'][unique_world_idx_gaze[j]:unique_world_idx_gaze[j + 1]]))
            temp_pos_x = np.stack(temp_pos_x)
            temp_pos_y = np.stack(temp_pos_y)
            gaze_speed = np.sqrt(
                np.square(temp_pos_x[0:-1] - temp_pos_x[1:]) + np.square(temp_pos_y[0:-1] - temp_pos_y[1:]))
            gaze_speed = list(gaze_speed)
            idx = 0


            # if not os.path.isdir(folder+id_name + '/Eye Tracker/Task 2/Trial ' + str(k+1)):
            #     os.makedirs(folder+id_name + '/Eye Tracker/Task 2/Trial ' + str(k+1))
            #
            # np.savetxt(folder+id_name + '/Eye Tracker/Task 2/Trial ' + str(k+1) + '/pupil_diam.txt',
            #            np.stack(temp))
            # # np.savetxt(folder + id_name + 'Eye Tracker/Task 2/Trial ' + str(k) + '/fixation.txt',
            # #            np.stack(temp))
            # np.savetxt(folder + id_name + '/Eye Tracker/Task 2/Trial ' + str(k+1) + '/gaze_speed.txt',
            #            np.stack(gaze_speed))

            print('Finished Trial ' + str(k+1) + ' !!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    print('Finished '+id_name+' !!!!!!!!!!!!!!!!!!!!!!!!!!!!')