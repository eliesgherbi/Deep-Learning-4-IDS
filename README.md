# Empirical-Evaluation-Of-In-Vehicle-IntrusionDetection-System-using-Deep-Learning  (The code will be available after the proceeding announcemnt)
Modern and future vehicles are complex cyber-physical sys-tems. The connection to their outside environment raises many securityproblems that impact our safety directly. In this work, we propose a DeepCAN intrusion detection system framework. We propose a multivariatetime  series  representation  for  asynchronous  CAN  data.  This  represen-tation  enhances  the  temporal  modelling  of  deep  learning  architecturesfor  anomaly  detection.  We  study  different  deep  learning  tasks  (super-vised/unsupervised) and compare different architectures, to propose anin-vehicle intrusion detection system that fits constraints of memory andcomputational  power  of  the  in-vehicle  system.  The  proposed  intrusiondetection system is time window wise: any given time frame is labelledeither anomalous or normal. We conduct experiments with many types ofattacks on an in-vehicle CAN using SynCAn dataset. We show that oursystem yields good results and allow to detect different kinds of attacks.

## DATAset

The dataset is available at : https://github.com/etas/SynCAN.

Note that, u need to set the configuration of the data set that u wish to have (create the train and test data). and change the variable "root_dir" in maint.py.

### Prerequisites

tensorflow 2.0.
keras.


### Running Classification

```
python main.py (name of the folder containing all the TS datasets) (specific dataset containing train, test raws) classification_model_name _itr_1 True

```

### Running Encoders
```
python main.py (name of the folder containing all the TS datasets) (specific dataset containing train, test raws) encoder_model_name _itr_1 True

```

