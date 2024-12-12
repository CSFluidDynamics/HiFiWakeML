from __future__ import print_function

import yaml
import torch
import numpy as np
from scipy.io import loadmat
import time

from netCDF4 import Dataset
import floris.layout_visualization as layoutviz
from floris import (FlorisModel, ParallelFlorisModel)
from floris.flow_visualization import visualize_cut_plane


def Floris_WF():

    inputs=read_inputfile()

# Load Wind probability    
    Wspds,Wdir,pdw = loadWD('WindDensity.mat')
    yaw_angles = np.array(inputs['wind_farm']['yaw_angle'],ndmin=2)
    layout_x = np.array(inputs['wind_farm']['layout_x'],ndmin=2)
    layout_y = np.array(inputs['wind_farm']['layout_y'],ndmin=2)

    Wspds=Wspds+0.5
    Wdir=Wdir-2.5
    xt,yt=set_turbines(inputs,Wdir)

    hgrid = np.array(inputs['domain']['hgrid'])*1000



    Lx = inputs['domain']['lx']
    Ly = inputs['domain']['ly']
    Nx = inputs['domain']['nx']
    Ny = inputs['domain']['ny']

    #Vci = inputs['wind_farm']['cutInVel']
    Vci = 8.5
    Vco = 9.5
    Vci_dir = 178
    Vco_dir = 183
    #Vco = inputs['wind_farm']['cutOutVel']

    fmodel = FlorisModel(inputs['florisinput_file']);

    for ii in range(Wdir.size):

        if ( (Wdir[ii]>=Vci_dir) and (Wdir[ii]<=Vco_dir)):

            for jj in range(Wspds.size):
                #if ( (Wspds[jj]>=Vci) and (Wspds[jj]<=Vco) ):
                if ( (Wspds[jj]>=Vci) and (Wspds[jj]<=Vco)):

                    print('Wind speed: ', Wspds[jj], 'Wind Dir: ', Wdir[ii])
                    print('Turbine Location ', xt[ii,:], yt[ii,:], Lx, Ly )
                    fmodel.set(layout_x = xt[ii,:], 
                               layout_y=yt[ii,:], 
                               wind_speeds=[Wspds[jj]],
                               yaw_angles=yaw_angles)
       
                    u =[]
                    for h in hgrid:
                        print('H',h)
                        hplane = fmodel.calculate_horizontal_plane(height=h, 
                                                                   x_bounds=(960,Lx),
                                                                   y_bounds=(2400,Ly),
                                                                   x_resolution=Nx,
                                                                   y_resolution=Ny) #960 -lx/ 2400-ly
                        u.append(hplane.df.u.values.reshape(Ny,Nx))

                    filename = './FLORIS_Field/Velfield_'+str(Wspds[jj]).zfill(5)+'_'+str(Wdir[ii]).zfill(6)+'.nc'
                    wrt = write_nc(np.asarray(u),filename,hplane,hgrid,Nx,Ny);
                    np.save(filename, np.nan_to_num(u, nan=0.0))

               # AEP += fmodel.get_farm_power()*pdw[jj,ii]*dt;


    #AEP *= 1e-6;
    #print('AEP:', AEP, 'GW-h')
    #filename2 = './AEP.nc'
   # writeAEP_nc(AEP,filename2,xt,yt,Wdir)


def write_nc(uu,filename,hplane,hgrid,Nx,Ny):

    xx=hplane.df.x1.values.reshape(Ny,Nx)
    yy=hplane.df.x2.values.reshape(Ny,Nx)
    x = np.squeeze(xx[1,:])
    z = np.squeeze(yy[:,1])
    y = np.squeeze(hgrid)


    print('Writing: ', filename)
    Vel_out = Dataset(filename,'w')
    MX = Vel_out.createDimension("MX", uu.shape[2])
    MY = Vel_out.createDimension("MY", uu.shape[0])
    MZ = Vel_out.createDimension("MZ", uu.shape[1])
    X = Vel_out.createVariable("X", "f4",("MX"))
    Y = Vel_out.createVariable("Y", "f4",("MY"))
    Z = Vel_out.createVariable("Z", "f4",("MZ"))
    U = Vel_out.createVariable("U", "f4", ("MY","MZ","MX"))
    

    X[:]=x
    Y[:]=y
    Z[:]=z
    U[:,:,:]=uu

    Vel_out.close()

    return 0
    
def set_turbines(inputs,Wdir):
    # Assumes xo is  North   (0 degrees)
    Lx = inputs['domain']['lx']
    Ly = inputs['domain']['ly']
    nt = inputs['wind_farm']['num_turbs']
    xto= np.array(inputs['wind_farm']['layout_x'])
    yto= np.array(inputs['wind_farm']['layout_y'])

    xc = np.mean(xto)
    yc = np.mean(yto)

    xto = xto-xc
    yto = yto-yc


    i=0
    wn = Wdir.size
    xtn = np.zeros((wn,nt))
    ytn = xtn
    for wd in Wdir:
        csr,snr = np.cos(wd), np.sin(wd)
        rotated_coords = np.dot(np.array([[csr, -snr], [snr, csr]]), 
                                np.vstack((xto, yto)))
        xtn[i,:], ytn[i,:] = rotated_coords[0] + xc, rotated_coords[1] + yc
        i+=1

    return xtn,ytn

def loadWD(filename):
    dat = loadmat(filename)
    Wspds = dat['Wspds'][0]
    Wdir = dat['deg'][0]
    pdw = dat['pdw']
    return Wspds, Wdir, pdw

##def AnnualEnergyProduction(xt,yt,Wspds,Wdir,pdw,model):
##### This function computes the annual energy production of a wind farm for an specific configuration
####    # Zero degree configuration (Input) correspond to wind comming from north
##    
###    t11 = time.time();
##    aep = 0;
##    dt = 1/6; # Ten minute statistics converted to seconds
##
##    ii=0;
##    nd = len(Wdir);
##    ns = len(Wspds);
##    nx = len(xt);
##    xtn = np.zeros((nd,nx));
##    ytn = np.zeros((nd,nx));
##
##    for ws in Wspds:
##        jj=0;
##        for wd in Wdir:
##            xtn[jj,:],ytn[jj,:] = rotateWF(wd,xt,yt);
##            jj+=1;
##        ## Compute Power here
###        Power = WFPower_model(model,xtn,ytn,np.asarray(ws)); # Kilo-Watts
##        aep += np.sum(np.multiply(Power,pdw[ii][:])*dt); #kWh
##
##        ii+=1;
##
##    aep *= 1e-6; #GWh
##    
##    return aep

def rotateWF(deg, xt, yt):
    # Rotate wind farm by deg degrees
    xc, yc = maxL * 0.5, maxL * 0.5
    xtn, ytn = xt - xc, yt - yc

    rad = np.deg2rad(deg)
    csr, snr = np.cos(rad), np.sin(rad)

    rotated_coords = np.dot(np.array([[csr, -snr], [snr, csr]]), np.vstack((xtn, ytn)))

    xtn, ytn = rotated_coords[0] + xc, rotated_coords[1] + yc

    return xtn, ytn

def read_inputfile():
    with open('input_file.yaml', 'r') as file:
        inputs = yaml.safe_load(file)

    return inputs
            
if __name__ == '__main__':

    Floris_WF();
   # WF_Enhancement();
    
