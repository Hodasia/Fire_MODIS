#%%
import numpy as np
from matplotlib import pyplot as plt
dic = {819:5402, 954:6663, 302:5822}
for file, img in dic.items():
    data=np.load('/workplace/dataset/MODIS/valid/'+str(file)+'.npy')
    for i in range(6):
        plt.imshow(data[img,i])
        plt.colorbar()
        plt.show()

# %%
