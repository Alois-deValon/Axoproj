from init import *
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
from tqdm import tqdm
from time import sleep
import timeit
from mpl_toolkits import axes_grid1
from scipy.special import cbrt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import convolve2d
import time



def i_near(array,value):
	idx=np.argmin(np.abs(array-value))
	return idx


def convolve(ra,dec,vrad,array,beamx,beamy,beamv,pa):
    pix=abs(np.diff(ra)[0])
    dv=abs(np.diff(vrad)[0])
    if beamv!=0:
        sigmav=beamv/(dv*2*np.sqrt(2*np.log(2))) #vturb=0.5 ??
        array=gaussian_filter1d(array,sigmav,axis=2)
    if (beamx!=0) or (beamy!=0):
        major=beamy/(pix*2*np.sqrt(2*np.log(2)))
        minor=beamx/(pix*2*np.sqrt(2*np.log(2)))
        pa=pa*np.pi/180
        dim_psf=int(major*7)
        PSF=np.zeros((dim_psf,dim_psf))
        for i in range(dim_psf):
            for j in range(dim_psf):
                x1=(j-(int(dim_psf/2.0)))*np.cos(pa) +(i-(int(dim_psf/2.0)))*np.sin(pa)
                y1=(i-(int(dim_psf/2.0)))*np.cos(pa) -(j-(int(dim_psf/2.0)))*np.sin(pa)
                PSF[i,j]=np.exp(-(x1**2/(2*minor**2) + y1**2/(2*major**2)))
        PSF=PSF/np.amax(PSF)
        for i in tqdm(range(len(vrad))):
            array[:,:,i]=convolve2d(array[:,:,i],PSF,mode="same")
    return array
def variation_wiggling(wiggling_param,x,y,z,phi,vr,vz,vphi):
    vr_projx=vr*np.cos(phi)
    vr_projy=vr*np.sin(phi)

    vz_projz=vz

    vphi_projx=vphi*np.sin(phi)
    vphi_projy=vphi*np.cos(phi)

    #### Wiggling_effect
    if wiggling_param["angle"]!=0:
        angle_wigg=wiggling_param["angle"]*np.pi/180.0
        tau_wigg=wiggling_param["period"]*365*3600.0*24.0
        offset_phi=wiggling_param["phi0"]
        phi_wigg=(2*np.pi*z*1.5E8)/(tau_wigg*vz)+offset_phi

        z_wigg=-x*np.sin(angle_wigg)+z*np.cos(angle_wigg)
        x_wigg=x*np.cos(angle_wigg)*np.cos(phi_wigg)-y*np.sin(phi_wigg)+z*np.sin(angle_wigg)*np.cos(phi_wigg)
        y_wigg=x*np.cos(angle_wigg)*np.sin(phi_wigg)+y*np.cos(phi_wigg)+z*np.sin(angle_wigg)*np.sin(phi_wigg)


        vz_wiggz=vz_projz*np.cos(angle_wigg)
        vz_wiggy=vz_projz*np.sin(angle_wigg)*np.sin(phi_wigg)
        vz_wigg=[vz_wiggy,vz_wiggz]
        vr_wiggz=-vr_projx*np.sin(angle_wigg)
        vr_wiggy=vr_projx*np.cos(angle_wigg)*np.sin(phi_wigg)+vr_projy*np.cos(phi_wigg)
        vr_wigg=[vr_wiggy,vr_wiggz]

        vphi_wiggz=-vphi_projx*np.sin(angle_wigg)
        vphi_wiggy=vphi_projx*np.cos(angle_wigg)*np.sin(phi_wigg)+vphi_projy*np.cos(phi_wigg)
        vphi_wigg=[vphi_wiggy,vphi_wiggz]
        breakpoint()

    else:
        z_wigg=z
        x_wigg=x
        y_wigg=y

        vz_wigg=[0,vz_projz]

        vr_wigg=[vr_projy,0]
        vphi_wigg=[vphi_projy,0]


    return x_wigg,y_wigg,z_wigg,vz_wigg,vr_wigg,vphi_wigg
def variation_incl(x,y,z,incl,vr,vz,vphi):

    zp=-np.cos(incl)*y+np.sin(incl)*z
    xp=x

    ur=vr[0]*np.sin(incl)+vr[1]*np.cos(incl)
    uz=vz[0]*np.sin(incl)+vz[1]*np.cos(incl)
    urot=vphi[0]*np.sin(incl)+vphi[1]*np.cos(incl)
    vproj=-(ur+uz+urot)
    return(xp,zp,vproj)
def variation_pa(x,z,pa):
    xp=np.cos(pa)*x+np.sin(pa)*z
    zp=-np.sin(pa)*x+np.cos(pa)*z
    return(xp,zp)
def create_datacube(model_class,ra,dec,vrad,incl=90,pa=0,
                    wiggling_param={"angle":0 , "period": 0, "phi0":0},
                    convolution={"beamx": 0, "beamy": 0,"beamv":0, "pa":0}):

    min_pix=min(np.abs(np.diff(ra)[0]),np.abs(np.diff(dec)[0]))
    phi_1D,param=model_class.create_profile()
    ii=incl*np.pi/180
    pa=pa*np.pi/180
    console_width = os.get_terminal_size().columns

    shape_start=np.shape(param[:,:,0])
    r=np.tile(param[:,:,0],(len(phi_1D),1,1))
    z=np.tile(param[:,:,1],(len(phi_1D),1,1))
    vr=np.tile(param[:,:,2],(len(phi_1D),1,1))
    vz=np.tile(param[:,:,3],(len(phi_1D),1,1))
    vphi=np.tile(param[:,:,4],(len(phi_1D),1,1))
    dem=np.tile(param[:,:,5],(len(phi_1D),1,1))
    del param
    phi=np.broadcast_to(phi_1D[:,np.newaxis,np.newaxis],(len(phi_1D),shape_start[0],shape_start[1]))
    ##### 3D CREATION : OUTFLOW REFERENTIAL
    x=r*np.cos(phi)
    y=r*np.sin(phi)
    x,y,z,vz,vr,vphi=variation_wiggling(wiggling_param,x,y,z,phi,vr,vz,vphi)
    ##### Calcul de projection due a l'inclinaison
    del phi
    x,z,vproj=variation_incl(x,y,z,ii,vr,vz,vphi)

    del y,vr,vz,vphi
    x,z=variation_pa(x,z,pa)

    ##### Rotation par le pa
    x=x.flatten(order='F')
    z=z.flatten(order='F')
    vproj=vproj.flatten(order='F')
    dem=dem.flatten(order='F')

    deltaRA= ra[1]-ra[0]
    deltaDEC=dec[1]-dec[0]
    deltaVRAD=vrad[1]-vrad[0]
    cond_x=(x<=np.min(ra)-np.abs(deltaRA)*0.5) | (x>=np.max(ra)+np.abs(deltaRA)*0.5)
    cond_z=(z<=np.min(dec)-np.abs(deltaDEC)*0.5) | (z>=np.max(dec)+np.abs(deltaDEC)*0.5)
    cond_v=(vproj<=np.min(vrad)-np.abs(deltaVRAD)*0.5) | (vproj>=np.max(vrad)+np.abs(deltaVRAD)*0.5)
    dem=dem[~(cond_x | cond_z | cond_v)]
    x=x[~(cond_x | cond_z | cond_v)]
    z=z[~(cond_x | cond_z | cond_v)]
    vproj=vproj[~(cond_x | cond_z | cond_v)]
    z=np.round((z-dec[0])/deltaDEC).astype(int)
    x=np.round((x-ra[0])/deltaRA).astype(int)
    vproj=np.round((vproj-vrad[0])/deltaVRAD).astype(int)
    map_layer=np.zeros((len(dec),len(ra),len(vrad)))
    print(''.center(console_width, '='))
    print(('THE DATA IS '+str(z.size * 5 * z.itemsize//1E6)+' MB').center(console_width))
    print(('TRANSFORMING INTO PPV OF '+str(map_layer.size * map_layer.itemsize//1E6)+' MB').center(console_width))
    print(''.center(console_width, '='))
    for i in tqdm(range(len(z))):
        map_layer[z[i],x[i],vproj[i]]+=dem[i]
    map_layer=map_layer/np.max(map_layer)

    if (convolution["beamx"]>0 and convolution["beamy"]>0) or (convolution["beamv"]>0):
        print(('CONVOLUTION').center(console_width))

        map_layer=convolve(ra,dec,vrad,map_layer,**convolution)

    return map_layer

if __name__ == '__main__':
