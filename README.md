<h1 align="center"> 
COVEE: A Dataset for Cognitive Modeling with Video, Electroencephalography, and Eye Tracker
</h1>

<!-- ### Items available -->
### Updates
- [x] Paper
- [x] Database license agreement
- [x] Instruction to download
- [x] Dataloader code
- [ ] Database release

### Overview

COVEE is hosted at Purdue [Data Depot](https://www.rcac.purdue.edu/storage/depot) and distributed through [Globus](https://www.globus.org/).

### Reviewer data Access and download

A special account has been made for reviewers to access the dataset. Details can be found in the attached COVEE_Reviewer_Access.pdf

### Data Request Form

Request for access to COVEE by filling out the Google form [here](https://docs.google.com/forms/d/e/1FAIpQLScBLroHC9FJWcjAWpbjkTYWnmbnoPe5iEwxW05dTEpMw3wQog/viewform?usp=dialog)

### Data Access and download

Once approved, users can install and use GlobusConnectPersonal to access the data [here](https://www.globus.org/globus-connect-personal).
Instructions on using Globus to transfer files are [here](https://docs.globus.org/guides/tutorials/manage-files/transfer-files/).

### Structure

The directory structure of the dataset is as follows. 

```    
└── COVEE                               
    ├── src
    │    ├── label_generation
    │    ├── model_related
    │    ├── simple_dataset_creation
    │    ├── raw_data_processer
    │    └── research_program
    │    └── verb_generation
    │
    ├── simple_datasets
    │   ├── ISA
    │   ├── NASA-TLX
    │   └── Task 1
    │
    └── user_data
        ├── 1008719828
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
        │           ├── Facial Features
        │           ├── Face Landmarks 
        │           └── Eye Landmarks
        │ 
        ├── 1566954358
        │   └── ...
        ├── ...
        ├── ...
        ├── ...
        └── 9793764153 \23 subjects

```


