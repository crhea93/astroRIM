# ----- IMPORTS ----- #
import os
import sys
import random
import pickle
import numpy as np
import pandas as pd
from sherpa.astro.ui import *
import matplotlib.pyplot as plt
from scipy import interpolate
from tqdm import tqdm
from sherpa.astro.data import DataPHA
from sherpa.astro.io import read_arf, read_rmf, read_pha
import gc
def create(rmf, arf, redshift, temp, abundance, nH, exp_time, ct):
    rmf1=unpack_rmf(rmf)
    arf1=unpack_arf(arf)
    data = DataPHA(name='faked', channel=None, counts=None, exposure=10.)
    data.set_arf(read_arf(arf))
    data.set_rmf(read_rmf(rmf))
    #Set the source
    set_model('faked', xsphabs.abs1*xsapec.apec)
    apec.Redshift = redshift
    apec.kT = temp
    apec.Abundanc = abundance
    abs1.nH = nH
    fake_pha("faked", arf1, rmf1, exposure=exp_time)
    # Set background model
    set_bkg_model('faked', xsapec.bkgapec+get_model_component('abs1')*xsbremss.bkgbremm)
    bkgapec.kT = 0.18
    bkgapec.kT.freeze()
    bkgbremm.kT = 40.0
    bkgbremm.kT.freeze()
    fake_pha("faked", arf1, rmf1, exposure=exp_time)
    data = [get_data_plot("faked").x,
            get_data_plot("faked", recalc=True).y,
            get_error("faked")]  # Noise
    true_spec = [get_source_plot("faked", recalc=True).x,get_source_plot("faked", recalc=True).y]
    clean()
    reset()
    return data, true_spec

def create_synthetic_data_pt2():
    cluster_file = open('ClustersWithSyntheticData.txt', 'a')
    main_dir = sys.argv[1]
    cluster = sys.argv[2]
    cl_num = int(sys.argv[3])
    obsid = int(sys.argv[4])
    redshift_val = float(sys.argv[5])
    nH_val = float(sys.argv[6])
    num_spec = int(sys.argv[7])
    rmf_dir_base = sys.argv[8]
    rmf_dir = os.path.join(rmf_dir_base, str(obsid))
    # Collect arfs and rmfs
    rmfs = []
    arfs = []
    snrs = []
    for item in os.listdir(rmf_dir):
        if item.endswith("rmf"):
            rmfs.append(os.path.join(rmf_dir, item))
        elif item.endswith("arf"):
            arfs.append(os.path.join(rmf_dir, item))
        else:
            pass
    if rmfs:
        #rmf = os.path.join(main_dir, '%s/%s/repro/R_500.rmf'%(cluster, obsid))
        #arf = os.path.join(main_dir, '%s/%s/repro/R_500.arf'%(cluster, obsid))
        spectra = {}  # {ct: [[x,y, noise],rmf_num,redshift,temp,abundance,nH, SNR]}
        true_spec = {}  # {ct: true_spec_int_y} 
        # Step through and create num_spec spectra
        for i in tqdm(range(num_spec)):
            rmf = random.choice(rmfs)
            rmf_ind = rmfs.index(rmf)
            arf = arfs[rmf_ind]
            # Select xspec parameters
            redshift = redshift_val  # random.uniform(0, 0.01)  # redshift
            temp = random.uniform(0.5, 8.0)  # temperature in keV
            abundance = random.uniform(0.1, 1.)  # Metallicity abundance in Z_solar
            nH = nH_val  #  random.uniform(0.005, 0.005)
            exp_time = random.uniform(300, 1000)  # Sample SNR
            data, true_spec_int = create(rmf, arf, redshift, temp, abundance, nH, exp_time, i)
            spectra[i] = [data, rmf_ind, redshift, temp, abundance, nH]
            true_spec[i] = true_spec_int
        cluster_file.write('%s, %s\n'%(cluster, obsid))
        cluster_file.close()
        pickle.dump(spectra, open('SyntheticData/spectra_%s.pkl'%obsid, 'wb'))
        pickle.dump(true_spec, open('SyntheticData/true_%s.pkl'%obsid, 'wb'))
    else:
        print("%s has no response matrices."%cluster)
    
create_synthetic_data_pt2()
