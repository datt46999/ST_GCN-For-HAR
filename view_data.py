import pickle
import os 
import numpy as np



# load data
data_path = "../../data/ntu60_2d.pkl"
with open(data_path, 'rb') as f:
    dataF = pickle.load(f)
print(type(dataF))
print(dataF.keys())
slp = dataF['split']
ann = dataF['annotations']
print("\n")
print(slp.keys())
print([len(slp[x]) for x in slp])

print(ann[3].keys())
kp = ann[0]['keypoint']
print(kp.shape)


kps = ann[0]['keypoint_score']
print(kps.shape)
