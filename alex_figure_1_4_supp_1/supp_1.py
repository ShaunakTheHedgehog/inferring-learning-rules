import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
import os 
from one.api import ONE

one = ONE(silent=True)
one = ONE()
ROOT = one.cache_dir.joinpath('wittenlab/Subjects/fip_').as_posix()# Set folder where data was downloaded


def alf_loader(alfpath):
    data=pd.DataFrame()
    data['choice'] = np.load(alfpath+'/_ibl_trials.choice.npy')*-1
    data['feedbackType'] = np.load(alfpath+ '/_ibl_trials.feedbackType.npy')
    data['water'] = np.zeros(len(data.choice))
    data.loc[data.feedbackType==1,'water'] = 3
    data['rolling_choice'] = data['choice'].rolling(3).mean()
    data['response_times'] = np.load(alfpath+ '/_ibl_trials.response_times.npy') - np.load(alfpath+ '/_ibl_trials.goCueTrigger_times.npy')
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
    counter = 0
    for file in sorted(os.listdir(rootdir)):
        d = os.path.join(rootdir, file)
        if os.path.isdir(d):
            print(d)
            day_df = pd.DataFrame()
            for ses in sorted(os.listdir(d)):
                s = os.path.join(d, ses)
                if os.path.isdir(s):
                    if os.path.isfile(s+'/unconditionedA.flag') or \
                       os.path.isfile(s+'/unconditionedB.flag'):
                        print('pre-training')
                        continue
                    else:
                        try:
                            ses_df= alf_loader(s+'/alf')
                            day_df = pd.concat([day_df,ses_df])
                        except:
                            print('error '+s)
                            continue
            if len(day_df)>0:
                counter += 1
            day_df['day'] = counter
            mouse_df = pd.concat([mouse_df,day_df])
    return mouse_df

matplotlib.rcParams.update({'font.size': 6})
matplotlib.rcParams.update({'font.sans-serif':'Arial'})

YFP  = np.array([901,905,906,908,913,914]).astype(str)
CHRMINE = np.array([902,903,904,907,910,911,912]).astype(str)
CONTRA_IS_RIGHT_YFP = np.array([1,0,1,0,1,0])
CONTRA_IS_RIGHT_CHRMINE = np.array([1,0,0,1,0,1,1])

MICE = np.array(['16', '35', '30', '28', '37', '32', '41', '43', '36', '31', '14',
       '39', '29', '33', '40', '34', '13', '15', '27', '38', '26', '42', '913', '910', '912', '904', '901', '907', '905', '914', '902',
       '911', '903', '908', '906'])

alf_dataset = pd.DataFrame()
for mouse in MICE:
    mouse_path = ROOT + mouse
    mouse_df = mouse_data_loader(mouse_path)
    if np.isin(mouse, CHRMINE):
        mouse_df['cohort'] = 'chrmine'
        if CONTRA_IS_RIGHT_CHRMINE[np.where(CHRMINE==mouse)[0]]==0:
            mouse_df['choice'] = mouse_df['choice']*-1
            mouse_df['signed_contrast'] = mouse_df['signed_contrast']*-1
            mouse_df['contra_is_right']=0
        else:
            mouse_df['contra_is_right']=0
    elif np.isin(mouse, YFP):
        mouse_df['cohort'] = 'yfp'
        if CONTRA_IS_RIGHT_YFP[np.where(YFP==mouse)[0]]==0:
            mouse_df['choice'] = mouse_df['choice']*-1
            mouse_df['signed_contrast'] = mouse_df['signed_contrast']*-1
            mouse_df['contra_is_right']=0
        else:
            mouse_df['contra_is_right']=0
    else:
        mouse_df['cohort'] = 'learning' 
    mouse_df['mouse'] = mouse
    alf_dataset = pd.concat([alf_dataset,mouse_df])

pal = {
  'learning': 'dodgerblue',
  'chrmine': 'r',
  'yfp': 'k'
}

plt.rcParams['axes.linewidth'] = 0.5 
plt.rcParams['xtick.major.width'] = 0.5 
plt.rcParams['ytick.major.width'] = 0.5 
plt.rcParams['xtick.minor.width'] = 0.5 
plt.rcParams['ytick.minor.width'] = 0.5 


fig,ax = plt.subplots(1,3)
plt.sca(ax[0])
plotting_df = alf_dataset.groupby(['mouse','day', 'cohort']).count()[['response_times']].reset_index()
learning_plotting_df = plotting_df.loc[plotting_df['cohort']=='learning']
sns.lineplot(data=learning_plotting_df.loc[(learning_plotting_df['day']<21)],x='day',y='response_times', 
            color='gray', errorbar='se', linewidth = 0.5)
plt.xlabel('Training day')
plt.ylabel('Trials completed')
plt.sca(ax[1])
plotting_df = alf_dataset.groupby(['mouse','day', 'cohort']).sum()[['water']].reset_index()
learning_plotting_df = plotting_df.loc[plotting_df['cohort']=='learning']
sns.lineplot(data=learning_plotting_df.loc[(learning_plotting_df['day']<21)],x='day',y='water', 
            color='gray', errorbar='se', linewidth = 0.5)
sns.despine()
plt.xticks(np.arange(21),np.arange(21))
plt.ylabel('Water obtained (ul)')
plt.xlabel('Training day')
plt.sca(ax[2])
plotting_df = alf_dataset.groupby(['mouse','day', 'cohort']).mean()[['response_times']].reset_index()
learning_plotting_df = plotting_df.loc[plotting_df['cohort']=='learning']
sns.lineplot(data=learning_plotting_df.loc[(learning_plotting_df['day']<21)],x='day',y='response_times', 
            color='gray', errorbar='se', linewidth = 0.5)
sns.despine()
plt.xticks(np.arange(21),np.arange(21))
plt.ylabel('Decision latencies (s)')
plt.xlabel('Training day')



fig,ax = plt.subplots(1,2)
plt.sca(ax[0])
sns.histplot(data=alf_dataset.loc[(alf_dataset['day']<21) & (alf_dataset['cohort']=='learning')], x='response_times', hue='cohort', 
             stat='proportion', palette=pal, bins=np.arange(0,60,0.05), alpha=1)
plt.axvline(0.2,linestyle='dashed', color='k', linewidth = 0.5)
plt.xlim(0,2.5)
plt.ylim(0,0.125)
sns.despine()
plt.xlabel('Decision latencies (s)')
plt.ylabel('Fraction')
ax[0].get_legend().remove()
plt.sca(ax[1])
sns.lineplot(data=plotting_df.loc[(plotting_df['day']<21)],x='day',y='response_times',hue='cohort', palette=pal, errorbar='se', linewidth = 0.5)
sns.despine()
plt.xticks(np.arange(21),np.arange(21))
plt.axhline(0.2,linestyle='dashed', color='grey',linewidth = 0.5)
plt.ylabel('Decision latencies (s)')
plt.xlabel('Training day')

plt.show()