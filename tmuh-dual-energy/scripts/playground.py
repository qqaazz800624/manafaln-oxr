#%%
import json
import os
from glob import glob

import pydicom
from sklearn.model_selection import KFold
from tqdm import tqdm
#%%

root = "/neodata/oxr/tmuh/label"
uids = os.listdir(root)
labels = glob(f'{root}/*.json')

#%%

# Count how many types of labels
# class = 6

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
# create a dictionary for each strokeColor and its corresponding annotation tag

tagpair = []
for i in range(len(labels)):
    with open(labels[i]) as f:
        data = json.load(f)
    shapes = data['shapes']
    for j in range(len(shapes)):
        tagpair.append([shapes[j].get('strokeColor'), shapes[j].get('tagAnno')])

tag_dict = {}
for tag in unique_class:
    taganno = []
    for i in range(len(tagpair)):
        if tagpair[i][0] == tag:
            taganno.append(tagpair[i][1])
    tag_dict[tag] = list(set(taganno))

tag_dict

#%%




