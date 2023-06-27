"""
Create synthetic data using response matrices

https://cxc.cfa.harvard.edu/sherpa/threads/fake_pha/


mean SNR and exposure time:
| exp_time | SNR |
|----------|-----|
| 1.0      | 11.6 |
| 5.0      | 26.9 |
| 10.0     | 37.2 |
| 100.0    | 123.1|
| 1000.0   | 385.1|
"""
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
# ----- VARIABLES ----- #
num_spec = 10000 # Number of spectra to create
main_dir = '/export/home/carterrhea/Documents/ChandraData/MachineLearningPaperI'
rmf_dir = os.path.join(main_dir, 'RMFs')
# ----- MAIN ----- #

def main():
    cluster_list = '../X-rayCatalog_V1_cleaned.csv'
    #Read in information
    mini_halos = pd.read_csv(cluster_list)
    names = mini_halos['Object Name']
    redshift_val = mini_halos['Redshift'].round(4)
    nH_val = mini_halos['n_H'].round(4)
    obsids_init = mini_halos['Chandra ObsID']
    #Step through each cluster
    cluster_file = open('ClustersWithSyntheticData.txt', 'w+')
    cluster_file.write('Cluster Name, ObsID\n')
    cluster_file.close()
    for cl_num, cluster in enumerate(names):
        cluster = cluster.replace(" ","")
        print("We are on cluster %s which is cluster number %i"%(cluster, cl_num))
        obsids_str = [i for i in obsids_init[cl_num].split(",")]  # Commas or spaces
        obsids = [int(ob) for ob in obsids_str]
        obsid = obsids[0] # Only the first obsid
        sys_command = "python create_syn_data_pt2.py %s %s %i %i %f %f %i %s"%(main_dir, cluster, cl_num, obsid, redshift_val.iloc[cl_num], nH_val.iloc[cl_num], num_spec, rmf_dir)
        os.system(sys_command)
    return None

main()
