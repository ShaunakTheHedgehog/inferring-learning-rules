import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib as mpl
from scipy.stats import zscore
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from one.api import ONE
import os
one = ONE(silent=True)
one = ONE()
ROOT =  one.cache_dir.joinpath('wittenlab/Subjects/fip_').as_posix()

def normalize(vector):
    return (vector-min(vector))/(max(vector)-min(vector))

def normalize1(vector):
    return (vector-min(vector))/((min(vector)*-1)-min(vector))

def alf_loader(alfpath):
    data=pd.DataFrame()
    data['choice'] = np.load(alfpath+'/_ibl_trials.choice.npy')*-1
    data['rolling_choice'] = data['choice'].rolling(3).mean()
    data['feedbackType'] = np.load(alfpath+ '/_ibl_trials.feedbackType.npy')
    data['response_times'] = np.load(alfpath+ '/_ibl_trials.response_times.npy') - np.load(alfpath+ '/_ibl_trials.goCueTrigger_times.npy')
    data['contrastRight'] = np.load(alfpath+ '/_ibl_trials.contrastRight.npy')
    data['contrastLeft'] = np.load(alfpath+ '/_ibl_trials.contrastLeft.npy')
    data['reaction_time'] = np.load(alfpath+ '/_ibl_trials.firstMovement_times.npy') - np.load(alfpath+ '/_ibl_trials.goCueTrigger_times.npy')
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
                if os.path.isdir(s):
                    if os.path.isfile(s+'/unconditionedA.flag') or \
                       os.path.isfile(s+'/unconditionedB.flag'):
                        print('pre-training')
                        print(s)
                        continue
                    else:
                        try:
                            print(s)
                            ses_df= alf_loader(s+'/alf')
                            ses_df['day'] = np.load(s + '/training_day.npy')
                            day_df = pd.concat([day_df,ses_df])
                        except:
                            print('error '+s)
                            continue          
            mouse_df = pd.concat([mouse_df,day_df])
    return mouse_df

# Set paths and load model
model = pd.read_json("choice_weight_means.json")
YFP  = np.array([901,905,906,908,913,914])
CHRMINE = np.array([902,903,904,907,910,911,912])
CONTRA_IS_RIGHT_YFP = np.array([1,0,1,0,1,0])
CONTRA_IS_RIGHT_CHRMINE = np.array([1,0,0,1,0,1,1])

model_results = pd.DataFrame()
for i, mouse in enumerate(model['mu'].keys()):
    mouse_model_results = pd.DataFrame()
    mouse_model_results['bias'] = model['mu'][mouse][0]
    mouse_model_results['right_weight'] = model['mu'][mouse][1]
    mouse_model_results['left_weight'] = model['mu'][mouse][2]
    mouse_model_results['choice_weight'] = model['mu'][mouse][3]
    mouse_model_results['bias_95'] = model['95conf'][mouse][0]
    mouse_model_results['right_weight_95'] = model['95conf'][mouse][1]
    mouse_model_results['left_weight_95'] = model['95conf'][mouse][2]
    mouse_model_results['choice_weight_95'] = model['95conf'][mouse][3]
    mouse_model_results['bias_5'] = model['5conf'][mouse][0]
    mouse_model_results['right_weight_5'] = model['5conf'][mouse][1]
    mouse_model_results['left_weight_5'] = model['5conf'][mouse][2]
    mouse_model_results['choice_weight_5'] = model['5conf'][mouse][3]
    mouse_model_results['contra_is_right'] = np.nan
    mouse_model_results['ses'] = np.arange(len(mouse_model_results))
    mouse_model_results['mouse'] = mouse
    if int(mouse)<100:
        mouse_model_results['cohort'] = 'Normal'
    else:
        if np.isin(int(mouse), YFP):
            mouse_model_results['cohort'] = 'YFP'
            mouse_model_results['contra_is_right'] = CONTRA_IS_RIGHT_YFP[np.where(YFP==int(mouse))][0]
        else:
            mouse_model_results['cohort'] = 'chrmine'
            mouse_model_results['contra_is_right'] = CONTRA_IS_RIGHT_CHRMINE[np.where(CHRMINE==int(mouse))][0]
    model_results = pd.concat([model_results,mouse_model_results])

model_results_learning = model_results.loc[model_results['cohort']=='Normal']
model_results_learning['ses'] = model_results_learning['ses']+1
model_results_learning = model_results_learning.loc[model_results_learning.ses<21]
model_results_learning['delta_weights'] = (model_results_learning.right_weight*-1) - model_results_learning.left_weight
model_results_learning['delta_weights_norm'] = (((model_results_learning.right_weight*-1) - model_results_learning.left_weight) / 
                                                 ((model_results_learning.right_weight*-1) + model_results_learning.left_weight))
model_results_learning.loc[(model_results_learning.right_weight*-1<0)|
                           (model_results_learning.left_weight<0), 'delta_weights_norm'
                           ] = np.nan
model_results_learning = model_results_learning.reset_index()

# Fig 1 a-b
dataset_photometry = pd.DataFrame()
for m in model_results_learning.mouse.unique():
    mouse = str(m)
    mouse_path = ROOT + mouse
    mouse_df = mouse_data_loader(mouse_path)
    if model_results_learning.loc[model_results_learning['mouse']==m,'contra_is_right'].mean()==0:
        mouse_df['contra_is_right']=0
        mouse_df['choice'] = mouse_df['choice']*-1
        mouse_df['signed_contrast'] = mouse_df['signed_contrast']*-1
    else:
        mouse_df['contra_is_right']=1
    mouse_df['cohort'] =  model_results_learning.loc[model_results_learning['mouse']==m,'cohort'].unique()[0]
    mouse_df['mouse'] = mouse
    dataset_photometry = pd.concat([dataset_photometry,mouse_df])

dataset_photometry['correct'] = 1*(dataset_photometry['feedbackType']>0)
dataset_photometry['contra'] = 1*(dataset_photometry['signed_contrast']>0)
dataset_photometry['choice_1'] = (dataset_photometry.choice>0)*1
dataset_photometry.loc[dataset_photometry['choice']==0,'choice_1'] = np.nan

#Remove no-choice trials
dataset_photometry = dataset_photometry.loc[dataset_photometry['choice']!=0]
dt = dataset_photometry.groupby(['mouse','day'])[['choice','correct', 'choice_1']].mean().reset_index()
dt = dt.loc[dt['day']<21]
learning_ranking = dt.loc[dt['day']>15].groupby(['mouse']).mean().sort_values('correct').index.to_numpy()
ranking_dict = dict(zip(learning_ranking, np.arange(len(learning_ranking))+1))
dt['rank']  = dt.mouse.map(ranking_dict)

#change blues colormap
cmap = mpl.cm.Blues(np.linspace(0,1,20))
cmap = mpl.colors.ListedColormap(cmap[10:,:-1])

fig, ax = plt.subplots(1,2)
plt.sca(ax[0])
sns.lineplot(data=dt, x='day',y='correct', hue='rank', palette=cmap)
plt.xlabel('Training Day')
plt.ylabel('Accuracy')
sns.despine()
plt.xlim(0,20)
ax[0].get_legend().remove()
plt.sca(ax[1])
sns.lineplot(data=dt, x='day',y='choice', hue='rank', palette=cmap)
plt.xlabel('Training Day')
plt.ylabel('Choice Asymmetry')
plt.xlim(0,20)
sns.despine()
ax[1].get_legend().remove()

# 1-f example mice
MICE = [27,35,37]
fig, ax = plt.subplots(3,1, sharex=True)
for i,mouse in enumerate(MICE):
    plt.sca(ax[i])
    mouse_data = model_results_learning.loc[model_results_learning['mouse']==mouse]
    mouse_data = mouse_data.sort_values('ses')
    plt.plot(mouse_data.ses, mouse_data.left_weight*-1, color='#1BC47C',linewidth=0.5, marker='.')
    plt.fill_between(mouse_data.ses, mouse_data.left_weight_5*-1, mouse_data.left_weight_95*-1, color='#1BC47C', alpha=0.2)
    plt.plot(mouse_data.ses, mouse_data.choice_weight, color='#D62027',linewidth=0.5, marker='.')
    plt.fill_between(mouse_data.ses, mouse_data.choice_weight_5, mouse_data.choice_weight_95, color='#D62027', alpha=0.2)
    plt.plot(mouse_data.ses, mouse_data.right_weight*-1, color='#B77FEA',linewidth=0.5,  marker='.')
    plt.fill_between(mouse_data.ses, mouse_data.right_weight_5*-1, mouse_data.right_weight_95*-1, color='#B77FEA', alpha=0.2)
    plt.plot(mouse_data.ses, mouse_data.bias*-1, color='#000000',linewidth=0.5,  marker='.')
    plt.fill_between(mouse_data.ses, mouse_data.bias_5*-1, mouse_data.bias_95*-1, color='#000000', alpha=0.2)
    plt.xticks(np.arange(21),np.arange(21))
    plt.axvline(x=1, color='grey', ls='dashed',linewidth=0.5)
    plt.axvline(x=10, color='grey', ls='dashed', linewidth=0.5)
    plt.axvline(x=20, color='grey', ls='dashed',linewidth=0.5)
    plt.axhline(y=0, color='k',ls='dashed',linewidth=0.5)
    plt.xlabel('Training Day')
    plt.ylabel('Behavioral weights')
    plt.ylim(-17,17)
    sns.despine()

early_data = model_results_learning.loc[(model_results_learning['ses']>0) & (model_results_learning['ses']<6)]
order = early_data.groupby(['mouse'])['bias'].mean().sort_values().index
asort = early_data.groupby(['mouse'])['bias'].mean().to_numpy().argsort()
init_bias = early_data.groupby(['mouse'])['bias'].mean().sort_values().to_numpy()

late_data = model_results_learning.loc[(model_results_learning['ses']>15) & (model_results_learning['ses']<21)]
late_right = late_data.groupby(['mouse'])['right_weight'].mean().to_numpy()[asort]
late_left = late_data.groupby(['mouse'])['left_weight'].mean().to_numpy()[asort]
late_bias = late_data.groupby(['mouse'])['bias'].mean().to_numpy()[asort]
late_data['contrast_mod'] = late_data.right_weight*-1 - late_data.left_weight
late_contrast_modulation = late_data.groupby(['mouse'])['contrast_mod'].mean().to_numpy()[asort]
init_bias = init_bias*-1
mouse_name = early_data.groupby(['mouse'])['bias'].mean().sort_values().index
clusters = np.zeros(len(mouse_name))
clusters[np.where(init_bias<-1)]=1
clusters[np.where(init_bias>1)]=2 
clust_dic = dict(zip(mouse_name, clusters))
model_results_learning['cluster'] = [clust_dic[m] for m in model_results_learning.mouse.to_list()]

# 1g - bias classification
fig, ax = plt.subplots()
sns.swarmplot(data=init_bias, color='k')
plt.axhline(y=1, color='gray',ls='dashed',linewidth=0.5)
plt.axhline(y=-1, color='gray',ls='dashed',linewidth=0.5)
plt.ylim(-3.5,3.5)
plt.tick_params(bottom=False)  # remove the ticks
plt.ylim(-3.5,3.5)
sns.despine()
plt.ylabel('Mean w Bias (Days 1-5)')

# 1h-j - bias clusters
fig, ax = plt.subplots(1,3)
plt.sca(ax[0])
sns.lineplot(data= model_results_learning.loc[model_results_learning.cluster==0], x='ses', y=model_results_learning.right_weight * -1,errorbar='se', color='#B77FEA')
sns.lineplot(data= model_results_learning.loc[model_results_learning.cluster==0], x='ses', y=model_results_learning.left_weight * -1,errorbar='se', color='#1BC47C')
sns.lineplot(data= model_results_learning.loc[model_results_learning.cluster==0], x='ses', y=model_results_learning.bias*-1,errorbar='se', color='k')
sns.lineplot(data= model_results_learning.loc[model_results_learning.cluster==0], x='ses', y=model_results_learning.choice_weight,errorbar='se', color='#D62027')
plt.xlim(0,20)
plt.ylim(-20,20)
plt.axvline(x=1, color='grey', ls='dashed',linewidth=0.5)
plt.axvline(x=10, color='grey', ls='dashed', linewidth=0.5)
plt.axvline(x=20, color='grey', ls='dashed',linewidth=0.5)
plt.axhline(y=0, color='k',ls='dashed',linewidth=0.5)
plt.title('Init weak bias (n=7)')
plt.ylabel('Weight')
plt.xlabel('Training day')
plt.xticks([1,10,20],[1,10,20])
sns.despine()
plt.sca(ax[1])
sns.lineplot(data= model_results_learning.loc[model_results_learning.cluster==2], x='ses', y=model_results_learning.right_weight * -1,errorbar='se', color='#B77FEA')
sns.lineplot(data= model_results_learning.loc[model_results_learning.cluster==2], x='ses', y=model_results_learning.left_weight * -1,errorbar='se', color='#1BC47C')
sns.lineplot(data= model_results_learning.loc[model_results_learning.cluster==2], x='ses', y=model_results_learning.bias*-1,errorbar='se', color='k')
sns.lineplot(data= model_results_learning.loc[model_results_learning.cluster==2], x='ses', y=model_results_learning.choice_weight,errorbar='se', color='#D62027')
plt.xlim(0,20)
plt.ylim(-20,20)
plt.axvline(x=1, color='grey', ls='dashed',linewidth=0.5)
plt.axvline(x=10, color='grey', ls='dashed', linewidth=0.5)
plt.axvline(x=20, color='grey', ls='dashed',linewidth=0.5)
plt.axhline(y=0, color='k',ls='dashed',linewidth=0.5)
plt.title('Init right bias (n=10)')
plt.ylabel('Weight')
plt.xlabel('Training day')
plt.xticks([1,10,20],[1,10,20])
sns.despine()
plt.sca(ax[2])
sns.lineplot(data= model_results_learning.loc[model_results_learning.cluster==1], x='ses', y=model_results_learning.right_weight * -1,errorbar='se', color='#B77FEA')
sns.lineplot(data= model_results_learning.loc[model_results_learning.cluster==1], x='ses', y=model_results_learning.left_weight * -1,errorbar='se', color='#1BC47C')
sns.lineplot(data= model_results_learning.loc[model_results_learning.cluster==1], x='ses', y=model_results_learning.bias*-1,errorbar='se', color='#000000')
sns.lineplot(data= model_results_learning.loc[model_results_learning.cluster==1], x='ses', y=model_results_learning.choice_weight,errorbar='se', color='#D62027')
plt.axvline(x=1, color='grey', ls='dashed',linewidth=0.5)
plt.axvline(x=10, color='grey', ls='dashed', linewidth=0.5)
plt.axvline(x=20, color='grey', ls='dashed',linewidth=0.5)
plt.axhline(y=0, color='k',ls='dashed',linewidth=0.5)
plt.xlim(0,20)
plt.ylim(-20,20)
plt.title('Init left bias (n=5)')
plt.ylabel('Weight')
plt.xlabel('Training day')
plt.xticks([1,10,20],[1,10,20])
sns.despine()

# 1l-k - bias correlations

fig,ax = plt.subplots(1,2)
plt.sca(ax[0])
plt.scatter(init_bias,late_bias, color='k')
plt.xlabel('Mean Bias ses 1-5')
plt.ylabel('Mean Bias ses 16-20')
plt.sca(ax[1])
plt.scatter(init_bias,late_contrast_modulation, color='k')
plt.xlabel('Mean Bias ses 1-5')
plt.ylabel('Mean R-L ses 16-20')
intercept = -0.697256 # From julia robust regression package - see methods
slope = 1.77155  # From julia robust regression package - see methods
x_vals = np.array(ax[1].get_xlim())
y_vals = intercept + slope * x_vals
plt.plot(x_vals, y_vals, color='k')
sns.despine()

plt.show()