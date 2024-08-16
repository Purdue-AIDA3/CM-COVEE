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
    ├── simple_datasets
    │   ├── Task 1   
    │   └── NASA-TLX
    │   
    ├── src
    │    ├── label_generation
    │    ├── model_related
    │    ├── simple_dataset_creation
    │    ├── raw_data_processer
    │    └── research_program
    │    └── verb_generation
    │
    └── user_data
        ├── 1008719828
        │   │   └── window_times.json
        │   │
        │   ├── click_data
        │   │   ├── clicks.json
        │   │   ├── ...
        │   │   ├── ...
        │   │   └── Task2_two_platforms_secondary.json
        │   │
        │   ├── Eye Tracker
        │   │   ├── Task 1
        │   │   └── Task 2
        │   │       ├── Trial 1
        │   │       ├── Trial 2
        │   │       ├── Trial 3
        │   │       └── Trial 4
        │   │
        │   ├── EEG
        │   │   ├── min_processed
        │   │   │   ├── log_ps
        │   │   │   ├── Task 1 With Secondary Task
        │   │   │   ├── Task 1 Without Secondary Task
        │   │   │   └── Task 2
        │   │   └── raw
        │   │       ├── Task 1 With Secondary Task
        │   │       ├── Task 1 Without Secondary Task
        │   │       └── Task 2
        │   │
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
        │           │   ├── Trial_1.mp4 
        │           │   ├── Trial_2.mp4
        │           │   ├── Trial_3.mp4
        │           │   └── Trial_4.mp4
        │           └── Facial Features
        │ 
        ├── 1566954358
        │   └── ...
        ├── ...
        ├── ...
        ├── ...
        └── 9793764153

```


