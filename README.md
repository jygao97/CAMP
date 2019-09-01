## pytorch implementation for the CAMP model in ICDM2019 paper "CAMP: Co-Attention Memory Networks for Diagnosis Prediction in Healthcare"
## Packages
- python 3.6
- torch 1.0.1
- numpy 1.15.4
## Run
- Step1: Create a folder named "data" and two sub-folders named "ccs" and "mimic". Then download the CCS-single-level file and CCS-multi-level file from [https://www.hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp](https://www.hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp) into the "ccs" folder. Download the mimic files "PATIENTS.csv", "DIAGNOSES _ICD.csv" and "ADMISSIONS.csv"  from [https://mimic.physionet.org/](https://mimic.physionet.org/) into the "mimic" folder.
- Step2: Use **processMIMIC.py** to extract patient historical records and demographics from MIMIC-III tables. After this step, we get "mimic.seqs", "mimic.profiles" and "mimic.pids" in the "mimic" folder.
- Step3: Use **preprocess.py** to split the dataset into three parts: training set, validation set and testing set. After this step, we get "mimic.train", "mimic.valid" and "mimic.test" in the "mimic" folder.
- Step4: Use **mkHierarchy.py** to extract the taxonomy from the CCS-multi-level file. After this step, we get "mimic.forgram"
in the "mimic" folder.
- Step5: With "mimic.train", "mimic.valid", "mimic.test", and "mimic.forgram" ready in the "mimic" folder, please use **run.py** to run our model, including training, validation in each epoch, test and evaluation.
