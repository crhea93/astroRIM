"""
This file will generate response files. To do this, we will use the CIAO function mkarf.
 This function needs an asphist file to make an ARF. In order to generate and asphist file, we will use the asphist command which takes the aspect solution file as an input.
 The important pieces of the aspect solution file (asol.fits) are the RA, DEC, and roll angle. Thankfully, these are in the header, so we can simply update these as we please!

Once we have this asphist file, we can pass it to mkarf while changing the sourcepixelx and sourcepixely variables.
The command will be something like this:

mkarf asphistfile="asphist.fits" outfile=acis_I3_arf.fits sourcepixelx=3950 sourcepixely=4900 engrid="0.1:10.0:0.01" dafile=NONE obsfile=evt2.fits detsubsys="ACIS-I3;CONTAM=NO"

where evt2.fits is some random event file we use.

Since the header we use really doesn't matter (well where we get the header and event file from that is),
 we will just take it from ObsID 7923


Carter Rhea
v0: October 28, 2021
"""

import os
import numpy as np
from astropy.io import fits
from tqdm import tqdm
from ciao_contrib.runtool import *
import pickle
import pandas as pd
from pylab import figure, cm
from matplotlib.colors import LogNorm
from matplotlib import pyplot as plt, cm
from matplotlib import colors
#from numba import jit, prange
from joblib import Parallel, delayed
def createRMFs(ct, rmf_dict, event_file, sourcepixelx_space, sourcepixely_space, obsid, RMF_Folder, rmf_ct):
    try:
        specextract.punlearn()
        specextract.infile = event_file+"[sky=circle(%i,%i,10)]"%((int(sourcepixelx_space[ct]), int(sourcepixely_space[ct])))
        #specextract.infile = event_file+"[sky=circle(%i,%i,10)]"%((int(sourcepixelx_space), int(sourcepixely_space)))
        specextract.outroot = RMF_Folder + '/' + obsid + '/' + str(rmf_ct)
        specextract.clobber = True
        specextract()
        # Create response matrix
        rmfimg.punlearn()
        rmfimg.infile = RMF_Folder + '/' + obsid + '/' + str(rmf_ct) + '.rmf'
        rmfimg.outfile = RMF_Folder +  '/' + obsid + '/' + str(rmf_ct) + '_rmf.img'
        rmfimg.arf = RMF_Folder + '/' + obsid + '/' + str(rmf_ct) + '.arf'
        rmfimg.arfout = RMF_Folder + '/' + obsid + '/'  + str(rmf_ct) + '_arf.img'
        rmfimg.product = True
        rmfimg.clobber = True
        rmfimg()
        # Get rmf and save in dictionary
        rmf_data = fits.open(RMF_Folder +  '/' + obsid + '/' + str(rmf_ct) + '_rmf.img')[0].data
        #arf_data = fits.open(RMF_Folder + '/' + obsid + '/' + str(ct) + '_arf.img')[0].data
        rmf_dict[rmf_ct] = rmf_data#[n_ignore_min:n_ignore_max, n_ignore_min:n_ignore_max]
        rmf_ct += 1
        #print(len(rmf_dict.keys()))
        #print(rmf_ct, end='')
        if rmf_ct%10 == 0:
            if not os.path.exists(RMF_Folder + '/ResponsePlots/%s'%obsid):
                os.makedirs(RMF_Folder + '/ResponsePlots/%s'%obsid)
            im= plt.imshow(rmf_data, cmap=cm.viridis)
            plt.ylabel('Detector channel in E-space')
            plt.xlabel('Detector channel in E"-space')
            cbar = plt.colorbar(im)
            cbar.set_label('Photon energy (keV)')
            plt.title('Response matrix %i'%rmf_ct)
            plt.savefig(RMF_Folder + '/ResponsePlots/%s/ResponseMatrix_%i.png'%(obsid, rmf_ct))
            plt.clf()
    except OSError:
        pass
    return rmf_ct


def main():
    outputPath = '/export/home/carterrhea/Documents/ChandraData/MachineLearningPaperI/RMFs'
    data_file = '/export/home/carterrhea/Documents/ChandraData/X-rayCatalog_V1_cleaned.csv'
    data = pd.read_csv(data_file)
    os.chdir('/export/home/carterrhea/Documents/ChandraData')

    with open('UncleanedObsIDs.txt', 'w+') as uncleanedFile:
        for index, row in data.iterrows():
            #try:
            objName = str(row['Object Name']).replace(" ", "")
            print(str(row['Chandra ObsID']).replace(" ", "").split(',')[0])
            obsid = str(row['Chandra ObsID']).replace(" ", "").split(',')[0]
            Obs_path = '/export/home/carterrhea/Documents/ChandraData/MachineLearningPaperI/%s/%s/primary/'%(objName, obsid)
            #evt_path = '/export/home/carterrhea/Documents/ChandraData/MachineLearningPaperI/%s/%s/repro/acisf%s_repro_evt2.fits'%(objName, obsid, obsid)
            evt_path = '/export/home/carterrhea/Documents/ChandraData/MachineLearningPaperI/%s/%s/repro/R_500.fits'%(objName, obsid)
            try:
                dmstat.punlearn()
                dmstat.infile="%s[bin sky=1]"%evt_path
                dmstat.centroid=True
                dmstat_ = dmstat()
                centroid = [float(val) for val in dmstat.out_cntrd_phys.split(',')]
                centroid_sigma = [float(val) for val in dmstat.out_sigma_cntrd.split(',')]
                if len(obsid) == 3:
                    obsid_asol = '00'+obsid
                elif len(obsid) == 4:
                    obsid_asol = '0'+obsid
                else:
                    obsid_asol = obsid
                asol_path = os.path.join(Obs_path, 'pcadf%s_000N001_asol1.fits'%obsid_asol)
                if not os.path.exists(asol_path):
                    asol_path = os.path.join(Obs_path, 'pcadf%s_000N002_asol1.fits'%obsid_asol)
                if not os.path.exists(asol_path):
                    asol_path = os.path.join(Obs_path, 'pcadf%s_000N003_asol1.fits'%obsid_asol)
                if not os.path.exists(asol_path):
                    asol_path = os.path.join(Obs_path, 'pcadf%s_000N004_asol1.fits'%obsid_asol)
                if not os.path.exists(asol_path):
                #Parallel(n_jobs=1, backend='multiprocessing')(delayed(createRMFs)(ct, rmf_dict, evt_path, sourcepixelx_space, sourcepixely_space, obsid,RMF_Folder) for ct in tqdm(range(n_samples)))
                    pass
                    #return False
                # Now we will step through our sampling of different values to create our resposne matrices
                n_samples = 50
                RMF_Folder = os.path.join(outputPath)
                if not os.path.exists(RMF_Folder):
                    os.makedirs(RMF_Folder)
                    print('Making folder')
                #ra_space = np.random.uniform(0, 45, n_samples)
                #dec_space = np.random.uniform(0, 45, n_samples)
                #roll_space = np.random.uniform(0, 20, n_samples)
                sourcepixelx_space = np.random.uniform(centroid[0] - 3*centroid_sigma[0], centroid[0] + 3*centroid_sigma[0], n_samples)
                sourcepixely_space = np.random.uniform(centroid[1] - 3*centroid_sigma[1], centroid[1] + 3*centroid_sigma[1], n_samples)
                rmf_dict = {}  # {id: rmf_matrix}
                rmf_ct = 0
                #resp = createRMFs(obsid[0], objName, outputPath)
                for ct in tqdm(range(n_samples)):
                #while rmf_ct < n_samples:
                    #sourcepixelx_space = np.random.uniform(3700, 4300)
                    #sourcepixely_space = np.random.uniform(3650, 4080)
                    rmf_ct = createRMFs(ct, rmf_dict, evt_path, sourcepixelx_space, sourcepixely_space, obsid, RMF_Folder, rmf_ct)
                    #print(rmf_ct)
                #Parallel(n_jobs=1, backend='multiprocessing')(delayed(createRMFs)(ct, rmf_dict, evt_path, sourcepixelx_space, sourcepixely_space, obsid,RMF_Folder) for ct in tqdm(range(n_samples)))
                pickle.dump(rmf_dict, open(os.path.join(outputPath,'rmfs_%s.pkl'%obsid), 'wb'))
                #if not resp:
                #    uncleanedFile.write("%s \n"%objName)
                #except FileNotFoundError as error:
                #    print(error)
                #    uncleanedFile.write("%s \n"%objName)
            except:
                print("%s does not have any R500 data"%objName)
            

main()
