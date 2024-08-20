import os
import pandas as pd
import time
import datetime
import json
import numpy as np
import glob


folder = '/COVEE/user_data/'

sub_folders = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]

print(sub_folders)

for i in range(1,len(sub_folders)):
    id_name = sub_folders[i]
    recording_location = folder+id_name+'/Eye Tracker/Task 1/000/'
    if os.path.isfile(recording_location+'info.player.json'):
        print(id_name)
        f = open(recording_location+'info.player.json')
        y = json.load(f)
        start_time_system = y["start_time_system_s"] # unix epoch timestamp
        start_time_synced = y["start_time_synced_s"] # pupil epoch timestamp
        offset = start_time_system - start_time_synced

        exported_pupil_csv = os.path.join(recording_location+'/exports/000/pupil_positions.csv')
        pupil_pd_frame = pd.read_csv(exported_pupil_csv)
        exported_gaze_csv = os.path.join(recording_location+'/exports/000/gaze_positions.csv')
        gaze_data = pd.read_csv(exported_gaze_csv)
        exported_fixation_csv = os.path.join(recording_location+'/exports/000/fixations.csv')
        fixation_data = pd.read_csv(exported_fixation_csv)

        clicks = glob.glob(folder+id_name+"/clicks_data/*.json")

        for video_num in range(0, np.shape(clicks)[0]):
            # video_num = 2
            if clicks[video_num][46:52] == 'Task_1':
                f_superbase = open(clicks[video_num])
                superbase_data = json.load(f_superbase)
                video_name = clicks[video_num][46:]
                print(video_name)
                video_start_timestamp = time.mktime(((datetime.datetime.strptime(superbase_data['video_start_time'],
                                                                                 "%Y-%m-%d %H:%M:%S.%f")) + datetime.timedelta(
                    hours=0)).timetuple())
                video_end_timestamp = time.mktime(((datetime.datetime.strptime(superbase_data['video_end_time'],
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


                fixation_start = next(x for x, val in enumerate(fixation_data['start_frame_index'])
                                            if val >= pupil_pd_frame['world_index'][idx_start])
                fixation_end = next(x for x, val in enumerate(fixation_data['start_frame_index'])
                                            if val >= pupil_pd_frame['world_index'][idx_end])-1

                if (pupil_pd_frame['world_index'][idx_end] - pupil_pd_frame['world_index'][idx_start])>=300:
                    temp_fixation_frame = np.zeros(
                        (pupil_pd_frame['world_index'][idx_end] - pupil_pd_frame['world_index'][idx_start],))
                else:
                    temp_fixation_frame = np.zeros((300,))

                for j in range(0,fixation_end-fixation_start):
                    temp_fixation_frame[(fixation_data['start_frame_index'][fixation_start+j]-pupil_pd_frame['world_index'][idx_start]):
                    (fixation_data['end_frame_index'][fixation_start+j]-pupil_pd_frame['world_index'][idx_start])] = 1

                temp = pupil_pd_frame[idx_start:idx_end + 1]
                video_pupil = temp[temp.method == 'pye3d 0.3.0 real-time']

                x, unique_world_idx = np.unique(video_pupil['world_index'], return_index=True)
                temp = []
                for j in range(0, -1 + np.shape(unique_world_idx)[0]):
                    temp.append(np.mean(video_pupil['diameter'][unique_world_idx[j]:unique_world_idx[j + 1]]))
                temp.append(np.mean(video_pupil['diameter'][unique_world_idx[-1]:]))
                idx = 0

                if (len(temp) < 300):
                    # print(video_num)
                    for frames in range(1, len(x) + idx):
                        # print(x[frames]-x[frames-1])
                        if (x[frames] - x[frames - 1]) != 1:
                            # print(x[frames])
                            place_holder = frames
                            for num_frames in range(0, x[frames] - x[frames - 1] - 1):
                                if (frames != 0):
                                    temp.insert(idx + place_holder - 1, 0.0)
                                else:
                                    temp.insert(idx + place_holder, 0.0)
                                idx += 1
                                frames += 1

                if len(temp) < 300:
                    for j in range(0, 300 - len(temp)):
                        temp.append(0.0)

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
                if (len(gaze_speed) < 300):
                    # print(video_num)
                    for frames in range(1, len(y) + idx):
                        # print(x[frames]-x[frames-1])
                        if (y[frames] - y[frames - 1]) != 1:
                            # print(x[frames])
                            place_holder = frames
                            for num_frames in range(0, y[frames] - y[frames - 1] - 1):
                                if (frames != 0):
                                    gaze_speed.insert(idx + place_holder - 1, 0.0)
                                else:
                                    gaze_speed.insert(idx + place_holder, 0.0)
                                idx += 1
                                frames += 1
                if len(gaze_speed) < 300:
                    for j in range(0, 300 - len(gaze_speed)):
                        gaze_speed.append(0.0)

                # if not os.path.isdir(folder+id_name+'/Eye Tracker/Task 1/ + 'gaze_speed'):
                #     os.makedirs(folder+id_name+'/Eye Tracker/Task 1/ + 'gaze_speed')
                # if not os.path.isdir(folder+id_name+'/Eye Tracker/Task 1/ + 'fixation'):
                #     os.makedirs(folder+id_name+'/Eye Tracker/Task 1/ + 'fixation')
                # if not os.path.isdir(folder+id_name+'/Eye Tracker/Task 1/ + 'pupil_diam'):
                #     os.makedirs(folder+id_name+'/Eye Tracker/Task 1/ + 'pupil_diam')
                #
                # np.savetxt(folder+id_name+'/Eye Tracker/Task 1/ + 'pupil_diam/'+clicks[video_num][46:-5]+'.txt', np.stack(temp[0:300]))
                # np.savetxt(recording_location + 'fixation/' + clicks[video_num][46:-5] + '.txt',
                #            np.stack(temp_fixation_frame[0:300]))
                # np.savetxt(recording_location + 'gaze_speed/' + clicks[video_num][46:-5] + '.txt',
                #            np.stack(gaze_speed[0:300]))

        print('Finished '+id_name+'!!!!!!!!!!!!!!!!!!!!!!!!!!!!')