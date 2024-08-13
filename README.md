<h1 align="center"> 
COVEE: A Dataset for Cognitive Modeling with Video, Electroencephalography, and Eye Tracker
</h1>

<!-- ### Items available -->
### Updates
- [ ] Paper
- [ ] Database license agreement
- [ ] Instruction to download
- [ ] Dataloader code
- [ ] Database release

### Overview

AVCAffe is hosted in [Globus](https://www.globus.org/).

The directory structure of the dataset is as follows. 

```    
    ├── ..                              
    ├── Simple Datasets
    │   └── Task 1   
    │       └── video_dataset_CL_112_30frames
    │           ├── 1008719828_Task_1_1_10_chunk_1.npz
    │           ├── ...
    │           ├── ...
    │           └── 9793764153_Task_1_5_9_chunk_8.npz
    │        
    ├── 1008719828
    │   │   ├── window_times.json
    │   │   ├── screen_recording.mp4    
    │   ├── click_data
    │   │   ├── clicks.json
    │   │   ├── ...
    │   │   ├── ...
    │   │   └── Task2_two_platforms_secondary.json
    │   ├── Eye Tracker
    │   │   ├── 000
    │   │   │   │   └── raw data files
    │   │   │   └── exports
    │   │   │       └── extracted data files
    │   │   ├── 001
    │   │   │   └── ...
    │   │   ├── 002
    │   │   │   └── ...
    │   │   ├── 003
    │   │   │   └── ...
    │   │   └── 004
    │   │       └── ...
    │   ├── EEG
    │   │   ├── min_processed
    │   │   │   ├── psd
    │   │   │   ├── Task 1 With Secondary Task
    │   │   │   ├── Task 1 Without Secondary Task
    │   │   │   └── Task 2
    │   │   └── raw
    │   │       ├── Task 1 With Secondary Task
    │   │       ├── Task 1 Without Secondary Task
    │   │       └── Task 2
    │   └── Video
    │       ├── Task 1
    │       │   │   ├── Task_1_1_1.mp4    # file names change per subject
    │       │   │   ├── ...
    │       │   │   ├── ...
    │       │   │   └── Task_1_1_1_secondary.mp4
    │       │   ├── Facial Features    # Facial features extracted from OpenFace
    │       │   ├── Face Landmarks 
    │       │   └── Eye Landmarks
    │       └── Task 2
    │           ├── Trial_1.mp4 
    │           ├── Trial_2.mp4
    │           ├── Trial_3.mp4
    │           └── Trial_4.mp4
    │ 
    ├── 1566954358
    │   └── ...
    ├── ...
    ├── ...
    ├── ...
    └── 9793764153

```


