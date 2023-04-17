#%%
import json
import os
from glob import glob

import pydicom
from sklearn.model_selection import KFold
from tqdm import tqdm
#%%

# Count how many types of labels
# class = 6

root = "/neodata/oxr/tmuh/label"
uids = os.listdir(root)
labels = glob(f'{root}/*.json')

strokeColor = []
for i in range(len(labels)):
    with open(labels[i]) as f:
        data = json.load(f)
    shapes = data['shapes']
    for j in range(len(shapes)):
        strokeColor.append(shapes[j]['strokeColor'])

unique_class = list(set(strokeColor))
print(unique_class)

#%%




