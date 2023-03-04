#%%
import os
import shutil
import SimpleITK as sitk  # noqa: N813
import numpy as np
import itk
import tempfile
import monai
from monai.data import PILReader
from monai.transforms import LoadImage, LoadImaged, Resized, Compose, SaveImage, EnsureChannelFirst
from monai.config import print_config

#print_config()


#%%

data_dir = '/home/u/qqaazz800624/manafaln-oxr/TMUH/data/image/1217243490150010'

filename = os.path.join(data_dir, '01846521.dcm')
loader = LoadImage(image_only=True, simple_keys=True)
channel_first = EnsureChannelFirst(channel_dim=-1)
data = channel_first(loader(filename))

print(f"image data shape: {data.shape}")
#print(f"meta data: {data.meta.keys()}")


#%%

filenames = ['/home/u/qqaazz800624/manafaln-oxr/TMUH/data/image/1217243490150010/01846520.dcm',
             '/home/u/qqaazz800624/manafaln-oxr/TMUH/data/image/1217243490150010/01846521.dcm',
             '/home/u/qqaazz800624/manafaln-oxr/TMUH/data/image/1217243490150010/01846522.dcm']

loader = LoadImage(image_only=False, simple_keys=True)
channel_first = EnsureChannelFirst(channel_dim=-1)

data = channel_first(loader(filenames))

print(f"image data shape: {data.shape}")
#print(f"{type(data)}")

#%%



#%%


data.shape



#%%





#%%





#%%






#%%






#%%





#%%






#%%