# --- path shim: allow running this legacy script standalone after the move to legacy/ ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end path shim ---
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from datetime import datetime

fips = [13, 14, 15, 16, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 901, 902, 903, 904, 905, 906, 907, 908, 910, 911, 912, 913, 914]


def get_session_paths(base_path, fip, stable_sub_path_str="alf"):
    """
    Constructs a DataFrame with paths and metadata for each session of a given mouse.
    """
    path_info = []
    data_path = os.path.join(base_path, f"fip_{fip}")
    if not os.path.isdir(data_path):
        print(f"Data path not found: {data_path}")
        return pd.DataFrame()
    
    dates = os.listdir(data_path)
    dates = [d for d in dates if os.path.isdir(os.path.join(data_path, d))]
    dates.sort(key=lambda d: datetime.strptime(d, "%Y-%m-%d"))

    # Iterate through session date folders (e.g., '2023-01-01')
    negative_counter = 0
    for date_dir in dates:
        op = os.path.join(data_path, date_dir)
        if not os.path.isdir(op):
            continue

        # Get numbered session attempt directories (e.g., '001')
        numbered_dirs = [d for d in os.listdir(op) if not d.startswith('.') and not d.startswith('training') ]
        
        if not numbered_dirs:
            continue
        
        # Use the first attempt directory
        session_dir = os.path.join(op, numbered_dirs[0])
        
        session_number = int(np.load(os.path.join(session_dir, "training_day.npy")).item())
        neural_missing = os.path.exists(os.path.join(session_dir, "no_neural.flag"))
        multiple_tries = len(numbered_dirs) > 1

        qc_dict = {"NAcc": np.nan, "DMS": np.nan, "DLS": np.nan}
        qc_path = os.path.join(session_dir, stable_sub_path_str, "FP_QC.json")

        if not neural_missing and os.path.exists(qc_path):
            with open(qc_path, 'r') as f:
                qc_dict = json.load(f)

        files_in_sessin_dir = os.listdir(session_dir)
        has_pre_training =  len([x for x in files_in_sessin_dir if "pre_training" in x]) > 0

        if has_pre_training and session_number == 1:
            negative_counter += 1

        path_info.append({
            "path": os.path.join(session_dir, stable_sub_path_str),
            "session": session_number - negative_counter,
            "multi": multiple_tries,
            "has_pre_training": has_pre_training,
            "neural_missing": neural_missing,
            "QC_NAcc": qc_dict.get("NAcc", np.nan),
            "QC_DMS": qc_dict.get("DMS", np.nan),
            "QC_DLS": qc_dict.get("DLS", np.nan)
        })

    return pd.DataFrame(path_info)

def get_trial_behavior(path, remove_miss_trials=True):
    """Loads behavioral data for each trial."""
    beh_names = [
        "_ibl_trials.goCueTrigger_times.npy",
        "_ibl_trials.firstMovement_times.npy",
        "_ibl_trials.feedback_times.npy",
        "_ibl_trials.feedbackType.npy",
        "_ibl_trials.choice.npy",
        "_ibl_trials.contrastLeft.npy",
        "_ibl_trials.contrastRight.npy"
    ]
    behavior_data_list = [np.load(os.path.join(path, name)) for name in beh_names]
    behavior_data = np.column_stack(behavior_data_list)

    trial_df =  pd.DataFrame({
        "stim_time": behavior_data[:, 0],
        "act_time": behavior_data[:, 1],
        "reward_time": behavior_data[:, 2],
        "reward_type": behavior_data[:, 3],
        "reward": (behavior_data[:, 3] + 1) / 2,  # maps raw reward -1, 1 to output idxs 0 (no reward), 1 (reward)
        "choice": behavior_data[:, 4],
        "y_choice": (behavior_data[:, 4] + 1) / 2,  # maps raw choices -1, 1 to output idxs 1 (go left), 0 (go right)
        "contrast_left": behavior_data[:, 5],
        "contrast_right": behavior_data[:, 6]
    })

    if remove_miss_trials:
        trial_df = trial_df[trial_df['choice'] != 0]
        trial_df = trial_df.reset_index(drop=True)

    # Fill NaN values in contrast columns with 0
    trial_df['contrast_left'] = trial_df['contrast_left'].fillna(0.)
    trial_df['contrast_right'] = trial_df['contrast_right'].fillna(0.)

    # swap the contrast columns to match choice encoding
    col1 = 'contrast_left'
    col2 = 'contrast_right'

    # Get the current column list
    cols = trial_df.columns.tolist()

    # Find the indices of the columns to swap
    idx1, idx2 = cols.index(col1), cols.index(col2)

    # Swap the column names in the list
    cols[idx1], cols[idx2] = cols[idx2], cols[idx1]

    # Reassign the DataFrame with the new column order
    trial_df = trial_df[cols]

    return trial_df

def get_all_trial_behavior(base_path, fip, stable_sub_path_str="alf"):
    """
    Compiles behavioral data across all sessions for a given mouse.
    """
    path_info = get_session_paths(base_path, fip, stable_sub_path_str)
    all_behavior = []

    for _, row in path_info.iterrows():
        session_behavior = get_trial_behavior(row['path'])
        session_behavior['session'] = row['session']
        all_behavior.append(session_behavior)

    all_behav = None

    if all_behavior:
        all_behav = pd.concat(all_behavior, ignore_index=True)
    else:
        all_behav = pd.DataFrame()

    inputs = all_behav[["contrast_right", "contrast_left"]].to_numpy()
    inputs = np.hstack((np.ones((inputs.shape[0], 1)), inputs))

    outputs = all_behav[["y_choice"]].to_numpy().flatten()

    rewards = all_behav[["reward"]].to_numpy().flatten()

    return all_behav, inputs, outputs, rewards

def get_all_mouse_behavior(base_path, fips, stable_sub_path_str="alf"):
    """
    Compiles behavioral data across all sessions for multiple mice.
    """
    full_behav_dict = {}

    for fip in fips:
        all_behav, inputs, outputs, rewards = get_all_trial_behavior(base_path, fip, stable_sub_path_str)
        full_behav_dict[fip] = {
            "behavior_df": all_behav,
            "inputs": inputs,
            "outputs": outputs,
            "rewards": rewards
        }
    return full_behav_dict



base_path = 'Subjects'
fip = 13
path_info = get_session_paths(base_path, fip, stable_sub_path_str="alf")

print(path_info)

behav_dict = get_all_mouse_behavior(base_path, fips, stable_sub_path_str="alf")

print(behav_dict)

np.save('all_mice_IBL_fips.npy', np.array(fips))

# with open('all_mice_IBL_behavioral_data.pkl', 'wb') as f:
#     pkl.dump(behav_dict, f)

