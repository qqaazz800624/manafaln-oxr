#%%
import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold


DICOM_DIR = "/neodata/oxr/innocare/dicom"

df = pd.read_csv(
    "/neodata/oxr/innocare/CAC_scores.csv", usecols=["UID", "鈣化指數"], index_col="UID"
)
uids = df.index

datalist = []
for uid in uids:
    # Add all images paths to data
    data = {
        channel: uid + "/" + channel + ".dcm"
        for channel in [
            "image_front_combined",
            "image_front_soft",
            "image_front_hard",
            "image_lateral_combined",
            "image_lateral_soft",
            "image_lateral_hard",
        ]
    }
    # Check if image path exists
    for path in data.values():
        path = os.path.join(DICOM_DIR, path)
        if not os.path.exists(path):
            raise ValueError(f"{path} not found")

    data["cac_score"] = df.loc[uid, "鈣化指數"]
    datalist.append(data)


#%%

k = 10 #number of folds
skf = StratifiedKFold(n_splits=k, random_state=42, shuffle=True)
k_fold_datalist = {}

for i, (train_idx, test_idx) in enumerate(skf.split(datalist, [data["cac_score"] > 400 for data in datalist])):
    test = [datalist[idx] for idx in test_idx]
    train_fold = [datalist[idx] for idx in train_idx]
    train, valid = train_test_split(train_fold, test_size=0.1,random_state=42,
                                         stratify=[data["cac_score"] > 400 for data in train_fold]
                                         )
    #k_fold_datalist[f'fold_{i}'] = {'train': train, 'valid': valid, 'test': test}
    k_fold_datalist[f'fold_{i}'] = test
   

#%%

with open("/neodata/oxr/innocare/datalist_b_cv.json", "w") as fp:
    json.dump(k_fold_datalist, fp, indent=4)
    

# %%
