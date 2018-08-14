"""Scratch file to play around with starburst99/mapping III grids from Dopita 2013
"""
import os 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from find_agn import notebook_utils

current_dir = os.path.dirname(os.path.realpath(__file__))
blue_grid_loc = current_dir + '/../data/dopita_2013_grid/apjs481187t4_blue_raw.txt'
red_grid_loc = current_dir + '/../data/dopita_2013_grid/apjs481187t5_red_raw.txt'

blue_names = [
    'z',    
    'kappa',
    'logq',
    'OII_3727', 
    'NII_3729',
    'NeIII_3869',
    'SII_4068',
    'Hg',
    'OIII_4363',
    'HeI_4471e',
    'HB_4861',
    'OIII_4959',
    'OIII_5006',  # originally 5007
    'HeI_5016e',
    'ArIII_5192',
    'NI_5198',
    'NII_5755'
]


red_names = [
    'z',
    'kappa',
    'logq',
    'HeI_5875',
    'OI_6300',
    'SIII_6313',
    'NII_6548',
    'HA_6562',
    'NII_6583',  # originaly 6584
    'HeI_6678',
    'SII_6716',  # originally 6716
    'SII_6731',
    'ArIII_7136',
    'OII_7318',
    'ArIII_775',
    'SIII_9068',
    'SIII_9532'
]

blue_df = pd.read_csv(blue_grid_loc, sep='\s+', names=blue_names, skiprows=30)
red_df = pd.read_csv(red_grid_loc, sep='\s+', names=red_names, skiprows=30)

assert len(blue_df) == len(red_df)
df = pd.merge(blue_df, red_df, how='inner', on=['z', 'kappa', 'logq'])
assert len(blue_df) == len(df)

df = notebook_utils.add_bpt_parameters(df)

sns.regplot(data=df, x='log_NII_HA', y='log_OIII_HB', fit_reg=False)
plt.savefig(current_dir + '/../figures/dopita_model_grid_bpt.png')