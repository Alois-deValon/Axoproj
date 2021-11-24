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
    console_width = os.get_terminal_size().columns
    obs=np.load("../plot_nature/obs_C5_6500_reb4_glb.npy",mmap_mode = "r")
    #vrad=np.linspace(7,20,200)
    dec=np.genfromtxt("../plot_nature/obs_DEC_C5_6500_reb4_glb.txt")*140
    ra=np.genfromtxt("../plot_nature/obs_RA_C5_6500_reb4_glb.txt")*140
    vrad=np.genfromtxt("../plot_nature/obs_VRAD_C5_6500_reb4_glb.txt")-6.34
    #vrad=np.arange(-5.1,5,0.3)
    print(''.center(console_width, '='))
    print()
    print('WELCOME TO  >(._.)<'.center(console_width))
    print(' AXOPROJ     (, ,)~ '.center(console_width))
    print()
    print(''.center(console_width, '='))




    model=DW_model(r0=3,theta=17,vp=15,J=20,alpha=1,I0=1)
    #model=WDS_model(C = np.array([0.02]), tau = np.array([500]),alpha = 0)

    model.create_setup(zmax=2000, step = 1, dphi = 1)
    #model.plot_dynamics()

    cube = create_datacube(model, ra, dec, vrad, incl = 90,pa=90,wiggling_param={"angle":3 , "period": 400, "phi0":0})#,convolution={"beamx": 20, "beamy": 20,"beamv":0, "pa":0})
    #np.save('../manuscrit/presentation/cube_cone.npy',cube)

    plt.imshow(np.sign(cube[:,:,i_near(vrad,0)]),origin='lower')
    plt.show()

    breakpoint()

    v=1
    ax=plt.gca()
    ax.imshow(cube[:,:,i_near(vrad,v)],origin='lower',cmap='inferno',vmax=0.7*np.max(cube[:,:,i_near(vrad,v)]))
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    breakpoint()




    #Channel_map_movie(ra,dec,vrad,cube,'test.mp4')
    v=2
    ax=plt.gca()
    ax.imshow(cube[:,:,i_near(vrad,v)],origin='lower',cmap='inferno')
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()
    breakpoint()
    #model.plot_dynamics(plot_J=True)

    levels_0=np.arange(-3*noise*1000,20*noise*1000,2)
    for i in tqdm(range(len(vrad))):
        fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(8,5))
        ax[1].imshow(data_conv[:,:,i],origin='lower',extent=(ra[0]/140,ra[-1]/140,dec[0]/140,dec[-1]/140),cmap='inferno')

        ax[0].imshow(obs[:,:,i]*1000,origin='lower',extent=(ra[0]/140,ra[-1]/140,dec[0]/140,dec[-1]/140),cmap='inferno',vmin=np.min(levels_0),vmax=np.max(levels_0))
        for j in range(2):
            ax[j].set_xlim(12,-12)
            ax[j].set_ylim(-5,19)
        plt.suptitle('V = '+str(np.round(vrad[i],1))+" km/s",fontsize=20)
        plt.savefig("cminfall/{0:0=2d}.png".format(i))
        plt.close()
    #duo_channel_map_movie(ra,dec,vrad,data_conv,obs,'Infall.mp4')

    breakpoint()


    time_vec=[]
    len_data=[]
    for dl_pixel in [5,1,0.5,0.1,0.05,0.01]:

        dec = np.arange(280,-280.1,-2)
        ra = np.arange(280,-280.01,-2)
        vrad=np.arange(-9.9,9.99,0.15)
        start = time.process_time()
        gamma=0.0
        disk=disk_model(rin=1,rout=182,Mstar=0.45,epsilon=0.3,alpha=1,side='both',gamma=gamma)
        disk.create_setup(step=dl_pixel,dphi=0.1)

        len_data.append(182/dl_pixel*2*3600)
        print(dl_pixel)
        #disk.plot_dynamics()
        #name="../accretion/models/inclinaison/85/disk"+str(gamma)
        data_conv=create_datacube(disk,ra,dec,vrad,incl=45,convolution={"beamx": 2*3, "beamy": 2*3,"beamv":0, "pa":0})
        #np.save(name,data_conv)
        end = time.process_time()

        #Channel_map_movie(ra,dec,vrad,data_conv,name+'.mp4')
        time_val = end - start
        time_vec.append(time_val)

    np.savetxt('temps_calcul.txt',np.vstack((len_data,time_vec)).T)
    breakpoint()

    """
    for theta_val in [20,40,60,80]:

        infall_data=Infall_model(rd=182,M=0.45,theta0=theta_val,J_sign=1)
        data_infall=create_datacube(infall_data,ra,dec,vrad,nb_per_pixel=3,dphi=0.1,incl=90)
        Channel_map_movie(ra,dec,vrad,data_infall,'disk_infall'+str(theta_val)+'.mp4')
    """
    """
    disk=disk_model(Mstar=0.8,rin=0.05,epsilon=0.3,rout=0.4,alpha=1,side="both")

    data_disk=create_datacube(disk,ra,dec,vrad,nb_per_pixel=3,dphi=1,incl=50)
    """

    """

    infall_data=Infall_model(rd=50,M=2,theta0=60)
    #infall_data.plot_dynamics(dl=1,zmax=300)

    data_infall=create_datacube(infall_data,ra,dec,vrad,nb_per_pixel=3,dphi=1,incl=50,zmax=100)

    Channel_map_movie(ra,dec,vrad,data_infall,'disk_infall.mp4')

    """


    r=np.array([0])
    z=np.array([100])
    vphi=np.array([1])
    vz=np.array([10])
    vr=np.array([5])
    I=np.array([1])

    test=Perso_model(r,z,vr,vz,vphi,I)
    #test.plot_dynamics(plot_J=False)
    #data_sconv =create_datacube(test,ra,dec,vrad,nb_per_pixel=10,dphi=0.1,incl=90)
    breakpoint()
    plt.imshow(data_conv[:,:,30],origin='lower')
    plt.show()

    """
    breakpoint()

    Channel_map_movie(ra,dec,vrad,datacube,'test_gaussian.mp4')
    """
    """
    C_para_cone=np.asarray([0.003,0.01,0.02,0.04])
    tau_cone=np.asarray([1600,850,500,220])
    eta_cone=np.asarray([0.5,0.5,0.5,0.5])

    test=WDS_model(C=C_para_cone,tau=tau_cone,eta=eta_cone)
    test.plot_dynamics(dl=10,zmax=2000,savename='hihi')
    """


    """

    vp=np.linspace(6,20,100)
    fit_polyr0=np.array([  0.18219106,  -5.33716296,  44.29352245, -73.26998141])
    fit_powtheta2=np.array([ 12.25634379, 139.70332814])
    fit_powtheta=np.array([ 12.63756189, 348.57164941,  -2.55223088])
    fit_vpJ=570
    r0_geo=polyn(fit_polyr0,vp)
    theta2=powerl2(vp,*fit_powtheta2)
    theta=powerl(vp,*fit_powtheta)
    #breakpoint()
    J=fit_vpJ/vp
    vz=vp*np.cos(theta)
    vr=vp*np.sin(theta)

    r0_geo[vp>13.5]=polyn(fit_polyr0,13.5)

    test=BS_model(vj=100,rho=1E4,zj=670,mdot=1E-8,v0=20,alpha=1)
    #test.plot_dynamics(dl=1,plot_J=False,savename='test_BS')
    """
    """
    test=Infall_model(rd=700,M=1.1,theta0=70)
    test.plot_dynamics(dl=10,zmax=3000)
    """
