#%%
import json
import os
from glob import glob

import pydicom
from sklearn.model_selection import KFold
from tqdm import tqdm


def getUID(json_path):
    with open(json_path, "r") as f:
        json_obj = json.load(f)
    return json_obj["accessionNumber"]

def getImagePath(uid):
    try:
        paths = {}
        for i in range(3):
            path = os.listdir(f"/neodata/oxr/tmuh/image/{uid}")[i]
            path = f"image/{uid}/{path}"
            ds = pydicom.dcmread(f"/neodata/oxr/tmuh/{path}")
            paths[ds.InstanceNumber] = path
    except FileNotFoundError as e:
        print(e)
        paths = None
    except IndexError as e:
        print(e)
        paths = None
    return paths

def main(n_splits=8):
    label_dir = "/neodata/oxr/tmuh/label/*.json"
    labels = glob(label_dir)
    labels = {
        getUID(label): label.replace("/neodata/oxr/tmuh/", "")
        for label in labels
    }
    images = {
        uid: getImagePath(uid)
        for uid in tqdm(labels)
    }
    datalist = [
        {
            "uid"  : uid,
            "image1": images[uid][1],
            "image2": images[uid][2],
            "image3": images[uid][3],
            "label": labels[uid]
        }
        for uid in labels
        if images[uid] is not None
    ]

    k_fold_datalist = {}

    for k, (_, fold) in enumerate(KFold(n_splits=n_splits, shuffle=True, random_state=42).split(datalist)):
        k_fold_datalist[f"fold_{k}"] = [datalist[i] for i in fold]

    with open("/neodata/oxr/tmuh/datalist_b.json", 'w') as fp:
        json.dump(k_fold_datalist, fp, indent=4)

if __name__ == "__main__":
    main()
# %%
