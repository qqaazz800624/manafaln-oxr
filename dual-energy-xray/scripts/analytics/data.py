import json

import matplotlib.pyplot as plt
import numpy as np

datalist_path = "data/neodata/datalist.json"

with open(datalist_path, "r") as f:
    datalist = json.load(f)

data = {}
for key, data_ in datalist.items():
    data[key] = [d["cac_score"] for d in data_]

data = [*data["train"], *data["valid"], *data["test"]]
data = np.array(data)

# Plot histograms for each dataset
plt.hist(data, bins=len(data))

# Add labels and titles
plt.xlabel("CAC score")
plt.ylabel("Count")
plt.title("CAC Distributions")

plt.savefig("distributions.png")
