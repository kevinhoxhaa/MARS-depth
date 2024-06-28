## This repository is part of the TU Delft course CSE3000 Research Project 2024/Q4.
The link for the research paper can be found [here](https://repository.tudelft.nl/islandora/object/uuid%3A745cde3c-b8e4-4f4e-8832-2a29745ba4ae?collection=education).
# MARS
**MARS: mmWave-based Assistive Rehabilitation System for Smart Healthcare**

_**This article appears as part of the ESWEEK-TECS special issue and will be presented in the International Conference on Hardware/Software Codesign and System Synthesis (CODES+ISSS), 2021. The proper publication information is in the reference section.**_

The figure below shows ten different movements we evaluate in our dataset.

They are:

_1) Left upper limb extension_  
_2) Right upper limb extension_  
_3) Both upper limb extension_  
_4) Left front lunge_  
_5) Right front lunge_  
_6) Squad_  
_7) Left side lunge_  
_8) Right side lunge_  
_9) Left limb extension_  
_10) Right limb extension_  

![overview](https://user-images.githubusercontent.com/82195094/114236867-dfb76d00-9947-11eb-90d5-130926828cbf.gif)

We also give an example demo of real-time joint angle estimation for _left front lunge_ movement from mmWave point cloud:

![LiveDemo](https://user-images.githubusercontent.com/82195094/115935697-5cbf0800-a459-11eb-9079-63a2c0b4dd34.gif)

**Dataset**

The folder structure is described as below.

```
${ROOT}
|-- synced_data
|   |-- wooutlier
|   |   |-- subject1
|   |   |   |-- timesplit
|   |   |-- subject2
|   |   |   |-- timesplit
|   |   |-- subject3
|   |   |   |-- timesplit
|   |   |-- subject4
|   |   |   |-- timesplit
|   |-- woutlier
|   |   |-- subject1
|   |   |   |-- timesplit
|   |   |-- subject2
|   |   |   |-- timesplit
|   |   |-- subject3
|   |   |   |-- timesplit
|   |   |-- subject4
|   |   |   |-- timesplit
|-- feature
|-- model
|   |-- Accuracy
```

**_synced_data_** folder contains all data with outlier/without outlier. Under the subject folder, there are synced kinect_data.mat and radar_data.mat if the readers want to play with individual movements. Under timesplit folder, there are train, validate, the test data and labels for each user. Note that labels here have all 25 joints from Kinect. In the paper, we only use 19 of them. Please refer to the paper for details of the 19 joints.

**_feature_** folder contains train, validate, the test feature and labels for all users. The features are generated from the synced data.

Dimension of the feature is (frames, 8, 8, 5). The final 5 means x, y, z-axis coordinates, Doppler velocity, and intensity.

Dimension of the label is (frames, 57). 57 means 19 coordinates in x, y, and z-axis. The order of the joints is shown in the paper.

**_model_** folder contains the pretrained model and and recorded accuracy.

**Dependencies**

- Keras 2.3.0
- Python 3.7
- Tensorflow 2.2.0


**Run the code**

The code contains load data, compile model, training, and testing. Readers can also choose to load the pretrained model and just do the testing.
```
python MARS_model.py
```
**License**

A MIT license is used for this repository. 

**Reference**
```
@article{10.1145/3477003,
author = {An, Sizhe and Ogras, Umit Y.},
title = {MARS: MmWave-Based Assistive Rehabilitation System for Smart Healthcare},
year = {2021},
issue_date = {October 2021},
volume = {20},
number = {5s},
issn = {1539-9087},
url = {https://doi.org/10.1145/3477003},
doi = {10.1145/3477003},
journal = {ACM Trans. Embed. Comput. Syst.},
month = sep,
articleno = {72},
numpages = {22},
keywords = {millimeter wave, Human pose estimation, point cloud, smart healthcare}
}


```
