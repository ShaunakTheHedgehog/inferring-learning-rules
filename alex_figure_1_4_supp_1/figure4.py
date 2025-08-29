import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib as mpl
from scipy.stats import zscore, ttest_rel, ttest_ind
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
import matplotlib
import matplotlib.ticker as mticker
import os 
from statsmodels.stats.multitest import fdrcorrection as fdr
from one.api import ONE
one = ONE(silent=True)
one = ONE()
ROOT =  one.cache_dir.joinpath('wittenlab/Subjects/fip_').as_posix()# Set folder where data was downloaded

def alf_loader(alfpath):
    data=pd.DataFrame()
    data['choice'] = np.load(alfpath+'/_ibl_trials.choice.npy')*-1
    data['rolling_choice'] = data['choice'].rolling(3).mean()
    data['feedbackType'] = np.load(alfpath+ '/_ibl_trials.feedbackType.npy')
    data['contrastRight'] = np.load(alfpath+ '/_ibl_trials.contrastRight.npy')
    data['contrastLeft'] = np.load(alfpath+ '/_ibl_trials.contrastLeft.npy')
    data.loc[np.isnan(data['contrastRight']), 'contrastRight'] = 0
    data.loc[np.isnan(data['contrastLeft']), 'contrastLeft'] = 0
    data['signed_contrast'] = data['contrastRight'] - data['contrastLeft']
    return data

def mouse_data_loader(rootdir):
    '''
    rootdir (str): mouse directory
    variables (list): list containing the keys of the variables of interest
    Will extract and load data from the whole life of animal
    '''
    mouse_df = pd.DataFrame()
    for file in sorted(os.listdir(rootdir)):
        d = os.path.join(rootdir, file)
        if os.path.isdir(d):
            day_df = pd.DataFrame()
            for ses in sorted(os.listdir(d)):
                s = os.path.join(d, ses)
                print(s)
                if os.path.isdir(s):
                    if os.path.isfile(s+'/unconditionedA.flag') or \
                       os.path.isfile(s+'/unconditionedB.flag'):
                        print('pre-training')
                        continue
                    else:
                        try:
                            ses_df= alf_loader(s+'/alf')
                            ses_df['day'] = np.load(s + '/training_day.npy')
                            day_df = pd.concat([day_df,ses_df])
                        except:
                            print('error '+s)
                            continue
            mouse_df = pd.concat([mouse_df,day_df])
    return mouse_df


matplotlib.rcParams.update({'font.size': 6})
matplotlib.rcParams.update({'font.sans-serif':'Arial'})

# Raw data
YFP  = np.array([901,905,906,908,913,914])
CHRMINE = np.array([902,903,904,907,910,911,912])
CONTRA_IS_RIGHT_YFP = np.array([1,0,1,0,1,0])
CONTRA_IS_RIGHT_CHRMINE = np.array([1,0,0,1,0,1,1])

dataset_photometry_opto = pd.DataFrame()
for i, mouse in enumerate(CHRMINE):
    mouse = str(mouse)
    mouse_path = ROOT + mouse
    mouse_df = mouse_data_loader(mouse_path)
    if CONTRA_IS_RIGHT_CHRMINE[i]==0:
        mouse_df['contra_is_right']=0
        mouse_df['choice'] = mouse_df['choice']*-1
        mouse_df['signed_contrast'] = mouse_df['signed_contrast']*-1
    else:
        mouse_df['contra_is_right']=1
    mouse_df['cohort'] = 'chrmine'
    mouse_df['mouse'] = mouse
    dataset_photometry_opto = pd.concat([dataset_photometry_opto,mouse_df])
for i, mouse in enumerate(YFP):
    mouse = str(mouse)
    mouse_path = ROOT + mouse
    mouse_df = mouse_data_loader(mouse_path)
    if CONTRA_IS_RIGHT_YFP[i]==0:
        mouse_df['contra_is_right']=0
        mouse_df['choice'] = mouse_df['choice']*-1
        mouse_df['signed_contrast'] = mouse_df['signed_contrast']*-1
    else:
        mouse_df['contra_is_right']=1
    mouse_df['cohort'] = 'YFP'
    mouse_df['mouse'] = mouse
    dataset_photometry_opto = pd.concat([dataset_photometry_opto,mouse_df])

dataset_photometry_opto['correct'] = 1*(dataset_photometry_opto['feedbackType']>0)
dataset_photometry_opto['contra'] = 1*(dataset_photometry_opto['signed_contrast']>0)
dataset_photometry_opto['choice_1'] = (dataset_photometry_opto.choice>0)*1
dataset_photometry_opto.loc[dataset_photometry_opto['choice']==0,'choice_1'] = np.nan
                            

dt = dataset_photometry_opto.groupby(['mouse','day','cohort', 'contra'])[['choice','correct', 'choice_1']].mean().reset_index()
dt = dt.loc[dt['day']<21]
dt['id'] = dt.mouse+dt.contra.astype(str)
dt['training_stage'] = 'late'
dt.loc[dt['day']<15, 'training_stage'] = 'middle'
dt.loc[dt['day']<7, 'training_stage'] = 'early'

fig, ax = plt.subplots(1,2)
plt.sca(ax[0])
sns.lineplot(data=dt.loc[dt['cohort']=='chrmine'], x='day', y='correct', hue='contra', errorbar='se',palette=['#ffc7c7','r'])
plt.ylabel('Accuracy')
plt.vlines(7,0,1, linestyles='dashed')
plt.vlines(15,0,1, linestyles='dashed')
plt.sca(ax[1])
sns.lineplot(data=dt.loc[dt['cohort']=='YFP'], x='day', y='correct', hue='contra', errorbar='se',palette=['grey','k'])
plt.ylabel('Accuracy')
plt.vlines(7,0,1, linestyles='dashed', color='k')
plt.vlines(15,0,1, linestyles='dashed', color='k')
sns.despine()

plt.show()