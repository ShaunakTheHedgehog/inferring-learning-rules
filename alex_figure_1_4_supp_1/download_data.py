from iblutil.util import Bunch
import numpy as np
import pandas as pd
from one.api import ONE
import os

one=ONE(base_url='https://openalyx.internationalbrainlab.org/', password='international')

def bleach_correct(nacc, avg_window=60, fr=25):
    """
    Correct for bleaching of gcamp across the session. Calculates
    DF/F
    Parameters
    ----------
    nacc: series with fluorescence data
    avg_window: time for sliding window to calculate F value in seconds
    fr: frame_rate
    """
    # First calculate sliding window
    avg_window = int(avg_window*fr)
    F = nacc.rolling(avg_window, center=True).mean()
    nacc_corrected = (nacc - F)/F
    return nacc_corrected

# Download and gerenerate trial data and associated files
paths  = np.load('training_ses.npy')
errors_trial_data = []
training_days_map = pd.read_parquet('all_subjects_training_day.pqt').reset_index()
pre_training_wheel_locked_dict = np.load('photometry_pre_training_unconditionedA.npy', allow_pickle=True)
pre_training_wheel_locked = []
for i in pre_training_wheel_locked_dict:
    for _, v in i.items():
        pre_training_wheel_locked.append(v)

pre_training_wheel_moving_dict = np.load('photometry_pre_training_unconditionedB.npy', allow_pickle=True)
pre_training_wheel_moving = []
for i in pre_training_wheel_moving_dict:
    for _, v in i.items():
        pre_training_wheel_moving.append(v)

for sess in paths:
    try:
        eid = one.path2eid(sess)
        trials = one.load_object(eid, 'trials')
        training_day = training_days_map.loc[training_days_map['eid']==eid,'training_day'].to_numpy()
        np.save(one.cache_dir.joinpath('wittenlab/Subjects/' + sess + '/alf/_ibl_trials.choice.npy'),trials.choice)
        np.save(one.cache_dir.joinpath('wittenlab/Subjects/' + sess + '/alf/_ibl_trials.contrastLeft.npy'),trials.contrastLeft)
        np.save(one.cache_dir.joinpath('wittenlab/Subjects/' + sess + '/alf/_ibl_trials.contrastRight.npy'),trials.contrastRight)
        np.save(one.cache_dir.joinpath('wittenlab/Subjects/' + sess + '/alf/_ibl_trials.feedback_times.npy'),trials.feedback_times)
        np.save(one.cache_dir.joinpath('wittenlab/Subjects/' + sess + '/alf/_ibl_trials.feedbackType.npy'),trials.feedbackType)
        np.save(one.cache_dir.joinpath('wittenlab/Subjects/' + sess + '/alf/_ibl_trials.firstMovement_times.npy'),trials.firstMovement_times)
        np.save(one.cache_dir.joinpath('wittenlab/Subjects/' + sess + '/alf/_ibl_trials.goCue_times.npy'),trials.goCue_times)
        np.save(one.cache_dir.joinpath('wittenlab/Subjects/' + sess + '/alf/_ibl_trials.goCueTrigger_times.npy'),trials.goCueTrigger_times)
        np.save(one.cache_dir.joinpath('wittenlab/Subjects/' + sess + '/alf/_ibl_trials.intervals.npy'),trials.intervals)
        np.save(one.cache_dir.joinpath('wittenlab/Subjects/' + sess + '/alf/_ibl_trials.response_times.npy'),trials.response_times)
        np.save(one.cache_dir.joinpath('wittenlab/Subjects/' + sess + '/alf/_ibl_trials.stimOnTrigger_times.npy'),trials.stimOnTrigger_times)
        np.save(one.cache_dir.joinpath('wittenlab/Subjects/' + sess + '/training_day.npy'), training_day[0])
        wheel = one.load_object(eid, 'wheel')
        np.save(one.cache_dir.joinpath('wittenlab/Subjects/' + sess + '/alf/_ibl_wheel.position.npy'),wheel.position)
        np.save(one.cache_dir.joinpath('wittenlab/Subjects/' + sess + '/alf/_ibl_wheel.timestamps.npy'),wheel.timestamps)
        if eid in pre_training_wheel_locked:
            f = open(one.cache_dir.joinpath('wittenlab/Subjects/' + sess + '/unconditionedA.flag'), 'x')
        if eid in pre_training_wheel_moving:
            f = open(one.cache_dir.joinpath('wittenlab/Subjects/' + sess + '/unconditionedB.flag'), 'x')
        np.save(one.cache_dir.joinpath('wittenlab/Subjects/' + sess + '/alf/_ibl_wheel.position.npy'),wheel.position)
        np.save(one.cache_dir.joinpath('wittenlab/Subjects/' + sess + '/alf/_ibl_wheel.timestamps.npy'),wheel.timestamps)
    except:
        errors_trial_data.append(sess)



# Download and gerenerate photometry data and associated files
paths  = np.load('photometry_ses.npy')
errors_photometry = []
for sess in paths:
    try:
        eid = one.path2eid(sess)
        fp = one.load_dataset(eid, 'photometry.signal.pqt')
        rois = one.load_dataset(eid, 'photometryROI.locations.pqt')
        fp = fp.loc[(fp.wavelength==470)&(fp.include==True)] #
        fp_corrected = Bunch()
        qc = Bunch()
        fr = (1 / (fp.times.diff().mean())).round()
        # Compute the bleach corrected signal for all regions and get the qc values from alyx for each region
        for roi, info in rois.iterrows():
            fp_corrected[info.brain_region] = bleach_correct(fp[roi], avg_window=60, fr=fr).values
            fiber = one.alyx.rest('insertions', 'list', session=eid, name=info.fiber)[0]
            qc[info.brain_region] = fiber['json']
        # Add the times for each frame to the dict
        fp_corrected['times'] = fp.times.values
        fp_corrected.times[np.where(np.isnan(fp_corrected.NAcc))] = np.nan
        np.save(one.cache_dir.joinpath('wittenlab/Subjects/' + sess + '/alf/_ibl_fluo.times.npy'),fp_corrected.times)
        np.save(one.cache_dir.joinpath('wittenlab/Subjects/' + sess + '/alf/_ibl_trials.NAcc.npy'),fp_corrected.NAcc)
        np.save(one.cache_dir.joinpath('wittenlab/Subjects/' + sess + '/alf/_ibl_trials.DLS.npy'),fp_corrected.DLS)
        np.save(one.cache_dir.joinpath('wittenlab/Subjects/' + sess + '/alf/_ibl_trials.DMS.npy'),fp_corrected.DMS)
        loc_dict = pd.Series({'Region2G':rois.loc[rois.index == 'Region2G'].brain_region[0],
                   'Region1G':rois.loc[rois.index == 'Region1G'].brain_region[0],
                   'Region0G':rois.loc[rois.index == 'Region0G'].brain_region[0]})
        loc_dict.to_json(one.cache_dir.joinpath('wittenlab/Subjects/' + sess + '/alf/loc_dict.json'))
        FP_QC = pd.Series({'DMS':1*(qc.DMS['qc']=='PASS'),
                   'NAcc':1*(qc.NAcc['qc']=='PASS'),
                   'DLS':1*(qc.DLS['qc']=='PASS')})
        FP_QC.to_json(one.cache_dir.joinpath('wittenlab/Subjects/' + sess + '/alf/FP_QC.json'))
    except:
        errors_photometry.append(sess)

# Add flags for no neural data sessions
paths_t  = np.load('training_ses.npy')
paths_n  = np.load('photometry_ses.npy')
no_neural = np.setxor1d(paths_t, paths_n)
for sess in no_neural:
    save_path = one.cache_dir.joinpath('wittenlab/Subjects/' + sess)
    save_path = os.path.join(save_path, 'no_neural.flag')         
    with open(save_path, 'w') as fp:
        pass


