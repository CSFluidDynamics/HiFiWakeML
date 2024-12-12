import numpy as np
import os
from netCDF4 import Dataset

path=["MHK1/", "MHK3/", "MHK4/"]

for i in range(0,3):
    all =[]
    for x in os.listdir(path[i]):
        if x.startswith("HY7"):
            all.append(x)
    all=sorted(all)
    for name in all:

        name_o=os.path.join(path[i],name)
        Vel=Dataset(name_o,'r')
        X=Vel.variables['X'][0:-1,0:-1,0:-1]#streamwise 880
        Y=Vel.variables['Y'][0:-1,0:-1,0:-1]#vertical 84
        Z=Vel.variables['Z'][0:-1,0:-1,0:-1]#spanwise 256

        W=Vel.variables['Vel_z'][1:-1,1:-1,1:-1].transpose(1,2,0)

        print("w shape", W.shape)
       
        
        Xc=(X[0:-1,0:-1,0:-1]+X[0:-1,0:-1,1:]\
            +X[0:-1,1:,0:-1]+X[0:-1,1:,1:]\
            +X[1:,0:-1,0:-1]+X[1:,0:-1,1:]\
            +X[1:,1:,0:-1]+X[1:,1:,1:])/8
        Yc=(Y[0:-1,0:-1,0:-1]+Y[0:-1,0:-1,1:]\
            +Y[0:-1,1:,0:-1]+Y[0:-1,1:,1:]\
            +Y[1:,0:-1,0:-1]+Y[1:,0:-1,1:]\
            +Y[1:,1:,0:-1]+Y[1:,1:,1:])/8
        Zc=(Z[0:-1,0:-1,0:-1]+Z[0:-1,0:-1,1:]\
            +Z[0:-1,1:,0:-1]+Z[0:-1,1:,1:]\
            +Z[1:,0:-1,0:-1]+Z[1:,0:-1,1:]\
            +Z[1:,1:,0:-1]+Z[1:,1:,1:])/8
            

            
        Vout=np.zeros((29, 1679, 476))
        Vout[:,:,:]=W[1:30,:,:]
        print(Vout.shape)


        filename_w=path[i]+"HY7.npy"
        np.save(filename_w, Vout)
        print("saved {}".format(filename_w))

