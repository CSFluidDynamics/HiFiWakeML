import numpy as np
import os
from netCDF4 import Dataset

path=["MHK1/", "MHK3/", "MHK4/"]

import numpy as np
import os
from netCDF4 import Dataset

path=["MHK1/", "MHK3/", "MHK4/"]

Vout=np.zeros((29, 1679, 476 )) #should be corrected - 29 - 1678 - 475
print(Vout.shape)

for i in range(0,3):
    all =[]
    for x in os.listdir(path[i]):
        if x.startswith("CY13"):
            all.append(x)
    all=sorted(all)
    for name in all:
        name_o=os.path.join(path[i],name)
        Vel=Dataset(name_o,'r')

        W=Vel.variables['U'][1:-1,1:-1,1:-1].transpose(1,2,0)
        print(W.shape)
        

        Vout[:,:,:]=W[:, :, :]/13
        #Vout[:,:,:]=W[:, :, :]/9.0

        print(Vout.shape)

        filename_w=path[i]+"CY13.npy"
        np.save(filename_w, Vout)
        print("saved {}".format(filename_w))
