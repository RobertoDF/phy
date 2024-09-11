from pathlib import Path
import subprocess
from utils.paths import path_to_trodes_export, server_folder
import numpy as np
from utils.settings import max_ISI_gap_recording
import torch
import sys
from spikeinterface.core import aggregate_units
import colorcet as cc
from spikeinterface.core import load_sorting_analyzer
from spikeinterface.extractors import read_kilosort, read_phy
from spikeinterface.extractors import read_spikegadgets
from tqdm import tqdm
from spikeinterface.core import get_noise_levels
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from utils.TrodesToPython.readTrodesExtractedDataFile3 import readTrodesExtractedDataFile
from scipy.io.matlab import loadmat
from matplotlib.patches import Rectangle
import seaborn as sns
import matplotlib.patches as mpatches
import panel as pn
from utils.settings import channel_label_color_dict, trodesexport_flags_to_folder,trial_extract_version
import spikeinterface.widgets as sw
import re
from datetime import time, date, datetime,  timedelta
from spikeinterface.core import set_global_job_kwargs
import shutil
from rich import print
import os
from spikeinterface.sorters import read_sorter_folder
from utils.settings import job_kwargs
from spikeinterface import create_sorting_analyzer
from spikeinterface.sorters import run_sorter
from spikeinterface.preprocessing import detect_bad_channels
from spikeinterface.extractors import read_kilosort
from spikeinterface.core import read_binary_folder
import pandas as pd
import torch
import shutil
import matplotlib.pyplot as plt
from probeinterface import write_prb
from kilosort import run_kilosort
from kilosort.io import load_probe
from spikeinterface.core import write_binary_recording
from utils.paths import server_folder
from pathlib import Path
from tqdm.auto import tqdm
from rich import print


set_global_job_kwargs(**job_kwargs)
print(f"set global job_kwargs: {job_kwargs}")

# Function to print in color

def camel_to_snake(name):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

def get_recording_time(path_recording_folder):
    # Convert the last part of the path (filename) to a string
    filename = path_recording_folder.name
    time_match = re.search(r'_(\d{2})(\d{2})(\d{2})', filename)
    hours, minutes, seconds = map(int, time_match.groups())  # Convert each group to integer
    # Create a datetime.time object
    extracted_time = time(hours, minutes, seconds)
    return extracted_time


def find_rec_file(directory):
    mouse_n = get_mouse_name(directory)
    day = get_recording_day(directory)

    try:
        time = get_recording_time(Path(directory))
    except:  # Replace with the specific exception if known
        time = get_recording_time(Path(*directory.parts[:-1]))

    print(f"mouse {mouse_n} recorded on {day} at {time}")
    rec_files = [f for f in directory.iterdir() if f.is_file() and f.name.endswith('.rec')]
    if len(rec_files) == 1:
        print(f"Exactly one .rec file found: {rec_files[0].name}")
        return rec_files[0], rec_files[0].name
    return "No .rec files found." if len(rec_files) == 0 else "More than one .rec file found"


def check_timestamps_gaps(times):
    intervals = np.diff(times)
    gap_indices = np.where(intervals > max_ISI_gap_recording)[0]
    gap_starts, gap_stops = times[gap_indices], times[gap_indices + 1]
    if len(gap_indices) > 0:
        for start, stop in zip(gap_starts, gap_stops):
            print(f"Gap from {start} to {stop}, duration {np.round(stop - start, 6)} s")
    else:
        print("No gaps detected.")
    return [gap_starts, gap_stops]

def check_overlap(trial_start, trial_stop, gap_starts, gap_stops):
    # Check if any gap start is less than the trial stop and any gap stop is more than the trial start
    return ((gap_starts < trial_stop) & (gap_stops > trial_start)).any()


def get_mouse_name(directory):
    if "ephys" in directory.parts:
        ephys_index = directory.parts.index("ephys")
        return directory.parts[ephys_index - 1] if ephys_index > 0 else None
    return None
def get_recording_day(directory):
    return directory.stem.split('_')[0]

def has_folder(directory, extension):
    for file_path in directory.iterdir():
        if file_path.is_dir() and file_path.name.endswith(extension):
            print( f"{extension} folder alread available: {file_path.name}")
            return True
    print(f"No folder with {extension} found.")
    return False

def call_trodesexport(path_recording_folder, flag):
    '''
    Extract flag info using trodesexport C++ executable
    '''
    if not has_folder(path_recording_folder, trodesexport_flags_to_folder[flag]):
        print(f"Extract {flag}")
        command = f"{path_to_trodes_export} -rec {path_recording_folder / path_recording_folder.parts[-1]} -{flag}"
        # Run the command
        try:
            if sys.platform == "win32":
                subprocess.run(command, check=True, shell=False, stdout=subprocess.PIPE, text=True)
            else:
                subprocess.run(command, check=True, shell=True, stdout=subprocess.PIPE, text=True)
            print("Command executed successfully")
        except subprocess.CalledProcessError as e:
            print("An error occurred while executing the command:", e)


def clean_trials(trials, gaps_start_stop, path_recording_folder):#TODO hit contains catch trials ==True
    '''
    Remove trials that contain gaps in recording,
    recalculates all time variables with absolute times
    converts to camel case
    saves a csv file
    '''

    trials = trials.copy()
    if len(gaps_start_stop[0]) > 0:
        trials['has_gap'] = trials.apply(lambda row: check_overlap(row['trial_start_time'], row['trial_stop_time'], gaps_start_stop[0], gaps_start_stop[1]), axis=1)
    else:
        trials['has_gap'] = False
    if trials['has_gap'].sum() > 0:
        print(f"Exclude {trials['has_gap'].sum()} trials because of recording gaps occurring within.")
        trials = trials.query('has_gap == False')
    else:
        print(f"No trials discarded")

    trials.drop(columns=["has_gap", "bpod_start_time"], inplace=True)
    trials.index.name = "trial_n"

    trials["ResponseTime"] = trials["ResponseTime"] + trials["StimulusStartTime"]# repsonse time is relative to stim start time! weird but true!

    for column in ["StimulusStartTime", "StimulusStopTime", "FeedbackDelay", "InvalidResponseTime",
                   "ResponseTime", "RewardStartTime", "WaitCin"]:
        if column in trials.columns:
            trials[column] = trials[column] + trials["trial_start_time"]

    trials['ChoiceRight'] = trials['ChoiceLeft'].apply(lambda x: True if x == 0 else False)
    trials['MadeChoice'] = trials['ChoiceLeft'].apply(lambda x: True if x == 0 or x == 1 else False)
    trials['Choice'] = trials['ChoiceLeft'].apply(lambda x: 'Left' if x == 1 else ('Right' if x == 0 else np.nan))

    if "Detection Confidence" in trials["stimulus_name"].unique(): # if new task add elif and func here
        trials = clean_detection_confidence_trials(trials)
    else:
        print("Task not present, add custom processing function if needed![white on red]")

    assert (trials["response_correct"] == (trials["trial_type"]=="hit") | (trials["trial_type"]=="cr")).all(), "Check here"# check categories make sense, response_correct comes from bpod

    trials["processing_version"] = trial_extract_version

    trials.to_csv(Path(f"{path_recording_folder}/trials.csv"))


def clean_detection_confidence_trials(trials):
    '''Processing related to detection confidence task only'''
    trials.columns = [camel_to_snake(column) for column in trials.columns]
    cols_to_move = ["trial_start_time", "trial_stop_time", "trial_duration"]
    trials[cols_to_move + [col for col in trials.columns if col not in cols_to_move]]

    assert trials.query("rewarded == 1 & embed_signal==1")["choice"].unique().shape[0]==1
    assert trials.query("rewarded == 1 & embed_signal==0")["choice"].unique().shape[0]==1
    poke_signal = trials.query("rewarded == 1 & embed_signal==1")["choice"].unique()[0]
    poke_no_signal = trials.query("rewarded == 1 & embed_signal==0")["choice"].unique()[0]


    trials['trial_type'] = np.select(
        [
            (trials['embed_signal'] == 1) & (trials['choice'] == poke_signal),  # hit
            (trials['embed_signal'] == 1) & (trials['choice'] == poke_no_signal),  # miss
            (trials['embed_signal'] == 0) & (trials['choice'] == poke_signal),  # fa
            (trials['embed_signal'] == 0) & (trials['choice'] == poke_no_signal)  # cr
        ],
        [
            'hit',  # Corresponds to the first condition
            'miss',  # Corresponds to the second condition
            'fa',  # Corresponds to the third condition
            'cr'  # Corresponds to the fourth condition
        ],
        default=pd.NA  # This will be used if none of the conditions are met
    )

    trials = optimize_trials_dtypes(trials)

    trials['correct_choice'] = trials['trial_type'].isin(["hit", "cr"])
    trials['skipped_reward'] = trials['correct_choice'] & ~trials['catch_trial'] & ~trials['rewarded']
    # Bin some vars
    signal_volume_bins = range(int(trials['signal_volume'].min()), int(trials['signal_volume'].max()),
                               int((trials['signal_volume'].max() - trials['signal_volume'].min()) / 10))

    trials['binned_signal_volume'] = pd.cut(trials['signal_volume'], bins=signal_volume_bins)

    trials['binned_waiting_time'] = pd.cut(trials['waiting_time'], bins=np.arange(0, 10, ))

    trials["early_withdrawal\n&\nbroke_fixation"] = trials[['early_withdrawal', 'broke_fixation']].any(axis=1)
    trials["leaving_time"] = trials["response_time"] + trials["waiting_time"]
    trials["error or catch"] = (trials['made_choice'] == True) & ((trials['response_correct']==False) | (trials['catch_trial'] == True) )

    return trials

def find_file(root_folder, target_file_name):
    found_files = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if target_file_name in filename:
                found_files.append(os.path.join(dirpath, filename))
    print(f"Found {len(found_files)} timestamps.dat files")
    return found_files


def get_timestamps_from_rec(path_recording_folder):
    path_timestamps = find_file(path_recording_folder, "timestamps.dat")
    if len(path_timestamps)==0:
        print("Extracting timestamps using trodesexport -time")
        call_trodesexport(path_recording_folder,"time")
        path_timestamps = find_file(path_recording_folder, "timestamps.dat")
    print(f"Read timestamps from {path_timestamps[0]}")
    timestamps_dict = readTrodesExtractedDataFile(path_timestamps[0])
    timestamps = timestamps_dict["data"]["time"]
    return timestamps/float(timestamps_dict["clockrate"]) # convert to0 time

def find_session_mat_files(bpod_path, path_recording_folder, raw_rec):
    '''
    Checks in bpod_session folder for a folder with the same date as the recording.
    Exclude files recorded outside the recording. Return files sorted by time.
    '''
    mouse_n = get_mouse_name(path_recording_folder)
    rec_time = get_recording_time(path_recording_folder)
    end_time_rec = (datetime.combine(date.today(),  rec_time) + timedelta(seconds=raw_rec.get_total_duration())).time() # we need a date to add times
    rec_date = get_recording_day(path_recording_folder)# day of rec
    bpod_directory = Path(bpod_path)
    mat_files = []
    for item in bpod_directory.iterdir():
        if item.is_dir():  # Ensure the item is a directory
            recording_day = get_recording_day(item)
            if recording_day == rec_date:
                for mat_file in item.glob('*.mat'):
                    print(f".mat file found: {mat_file}")
                    bpod_data = loadmat(mat_file, simplify_cells=True)['SessionData']
                    start_time_mat_file = datetime.strptime(bpod_data['Info']['SessionStartTime_UTC'], '%H:%M:%S').time()
                    assert mouse_n==bpod_data["Info"]["Subject"], "bpod file subject does not correspond to .rec file"

                    if (start_time_mat_file>rec_time) and (start_time_mat_file< end_time_rec):
                        print("Bpod file starts within the Trodes recording")
                        mat_files.append((mat_file, start_time_mat_file))
                    else:
                        print("Bpod file starts outside the Trodes recording! [white on red]")

    mat_files.sort(key=lambda x: x[1])  # x[1] is the time part of the tuple
    out = [file for file, _ in mat_files]  # Return only the file paths
    if len(out)>0:
        return out
    else:
        print("No bpod file found")


def check_gpu_availability():
    if torch.cuda.is_available():
        print(f"GPU available: n = {torch.cuda.device_count()}")
    else:
        "GPU not available"

def select_DIO_channel(path_DIO_folder):
    ''' Check if multiple DIO channels have info. Select the one with more than 10 pulses.
    '''
    DIO_with_data = []
    for file in os.listdir(path_DIO_folder):
        DIO_dict = readTrodesExtractedDataFile(Path(path_DIO_folder, file))
        if len(DIO_dict['data'])>10: # stupid euristic
            print(f"{file} contains data")
            DIO_with_data.append(DIO_dict)
    print(f"{len(DIO_with_data)} DIO files with data")
    assert len(DIO_with_data)==1, "No or multiple DIO files!"
    DIO_dict = DIO_with_data[0]

    return DIO_dict


def find_nearest_elements_by_diff(array_1, array_2):
    diff_array_1 = np.diff(array_1)
    diff_array_2 = np.diff(array_2)

    nearest_elements = []

    n = 0
    nn = 0
    offset = 0

    while n < len(diff_array_1) and (nn < len(diff_array_2)):

        distance = diff_array_2[nn + offset] - diff_array_1[n]

        if (abs(distance) < .2):
            nearest_elements.append((array_1[n], array_2[nn + offset]))
        else:
            distance = diff_array_2[nn + 1 + offset] - diff_array_1[n + 1]
            if abs(distance) < 0.2:
                nearest_elements.append((array_1[n], array_2[nn + offset]))
            else:
                distance = diff_array_2[nn + 1] - diff_array_1[n + 1]
                while (abs(distance) > 0.2):
                    offset += 1
                    if nn == len(diff_array_2):
                        break
                    distance = diff_array_2[nn + offset + 1] - diff_array_1[n + 1]

                nearest_elements.append((array_1[n], array_2[nn + offset]))
        n += 1
        nn += 1
    nearest_elements.append((array_1[n], array_2[nn + offset]))

    return pd.DataFrame(nearest_elements, columns=["bpod_start_time", "TTL_start_time"])


def correct_bpod_values(reconciled_bpod_TTLs):
    reconciled_bpod_TTLs["TTLdiff"] = reconciled_bpod_TTLs["TTL_start_time"].diff()
    for i in range(1, len(reconciled_bpod_TTLs)):
        # If the current bpod value is less than the previous one
        if reconciled_bpod_TTLs.loc[i, 'bpod_start_time'] < reconciled_bpod_TTLs.loc[i - 1, 'bpod_start_time']:
            # Calculate the new bpod values
            prev_value = reconciled_bpod_TTLs.loc[i - 1, 'bpod_start_time']
            diff_value = reconciled_bpod_TTLs.loc[i, 'TTLdiff'] - reconciled_bpod_TTLs.loc[i, 'bpod_start_time']

            # Update the current row and all subsequent rows
            for j in range(i, len(reconciled_bpod_TTLs)):
                reconciled_bpod_TTLs.loc[j, 'bpod_start_time'] = reconciled_bpod_TTLs.loc[
                                                                     j, 'bpod_start_time'] + prev_value + diff_value

    return reconciled_bpod_TTLs.drop("TTLdiff", axis=1)

def extract_trials(bpod_file, DIO_timestamps_start_trial, axs=None):
    trials_data_dfs = []
    for n, file in enumerate(bpod_file):
        print(file)
        bpod_data = loadmat(file, simplify_cells=True)['SessionData']
        print(f"Bpod session started at {bpod_data['Info']['SessionStartTime_UTC']},"
              f" duration: {bpod_data['TrialEndTimestamp'][-1] / 60} min, ended at: {(datetime.strptime(bpod_data['Info']['SessionStartTime_UTC'], '%H:%M:%S') + timedelta(minutes=bpod_data['TrialEndTimestamp'][-1] / 60)).strftime('%H:%M:%S')}")  # not used in calculations
        print(f"number trials: {len(bpod_data['TrialStartTimestamp'])}")

        if "AuditoryTuning" in str(file):
            print("Extracting AuditoryTuning params")
            TrialData_dict = {key: value for key, value in bpod_data["Custom"].items() if
                              key in ['Frequency', "Volume"]}
            trial_data_df = pd.DataFrame(TrialData_dict)
            trial_data_df["StimulusStartTime"] = pd.Series(
                [q["States"]["PlaySound"][0] for q in bpod_data["RawEvents"]["Trial"]])
            trial_data_df["StimulusStopTime"] = pd.Series(
                [q["States"]["PlaySound"][1] for q in bpod_data["RawEvents"]["Trial"]])
            trial_data_df["StimulusDuration"] = trial_data_df["StimulusStopTime"] - trial_data_df["StimulusStartTime"]

        elif "DetectionConfidence" in str(file):
            print("Extracting DetectionConfidence params")
            TrialData_dict = {key: value for key, value in bpod_data["Custom"]["TrialData"].items() if
                              not isinstance(value, int)}

            TrialData_dict = {key: value[:bpod_data["nTrials"]] for key, value in TrialData_dict.items() if
                              len(value) == bpod_data["nTrials"] or len(value) == bpod_data[
                                  "nTrials"] + 1}  # some fields have 1 row in excess
            trial_data_df = pd.DataFrame(TrialData_dict)
            trial_data_df["WaitCin"] = pd.Series([q["States"]['wait_Cin'][0] for q in bpod_data["RawEvents"]["Trial"]])
            trial_data_df.rename(columns={"StimDuration": "StimulusDuration"}, inplace=True)
            trial_data_df["StimulusStopTime"] = trial_data_df["StimulusStartTime"] + trial_data_df["StimulusDuration"]

        trial_data_df["stimulus_name"] = extract_protocol(bpod_data["Info"]["SessionProtocolBranchURL"].split("/")[-1])

        trial_data_df["bpod_start_time"] = bpod_data['TrialStartTimestamp']
        trial_data_df["trial_duration"] = bpod_data['TrialEndTimestamp'] - bpod_data['TrialStartTimestamp']

        trials_data_dfs.append(trial_data_df)

    trials = pd.concat([pd.concat(trials_data_dfs, ignore_index=True)])

    reconciled_bpod_TTLs = find_nearest_elements_by_diff(trials["bpod_start_time"], DIO_timestamps_start_trial)
    reconciled_bpod_TTLs = correct_bpod_values(reconciled_bpod_TTLs)

    trials.drop("bpod_start_time", axis=1, inplace=True)
    assert trials.shape[0] == reconciled_bpod_TTLs.shape[0]
    trials = pd.concat([trials, reconciled_bpod_TTLs], axis=1)
    trials.rename(columns={"TTL_start_time": "trial_start_time"}, inplace=True)
    trials["trial_stop_time"] = trials["trial_start_time"] + trials["trial_duration"]

    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(15,5))

    sns.scatterplot(data=trials, x="bpod_start_time", y="stimulus_name", s=5, ax=axs[0], hue="stimulus_name")

    sns.lineplot(data=trials, x="bpod_start_time", y="trial_start_time", ax=axs[1], hue="stimulus_name")
    for ax in axs.flatten():
        sns.despine(ax=ax)
    plt.tight_layout()

    return trials


def select_DIO_sync_trial_trace(path_recording_folder):
    ''' We select from the DIO trace containing TTL pulses only the trial starts (the ones)'''

    rec_file_name = path_recording_folder.parts[-1]
    path_DIO_folder = Path(path_recording_folder, f"{rec_file_name[:rec_file_name.rfind('.')]}.DIO")
    DIO_dict = select_DIO_channel(path_DIO_folder)
    # Each data point is (timestamp, state) -> break into separate arrays
    DIO_data = DIO_dict['data'].copy()
    DIO_states = np.array([tup[1] for tup in DIO_data])
    DIO_timestamps = np.array([tup[0] for tup in DIO_data])/float(DIO_dict['clockrate'])
    assert DIO_states.shape == DIO_timestamps.shape
    DIO_timestamps_start_trial = DIO_timestamps[DIO_states.astype(bool)].copy()# isolate start trial times 0>1

    return DIO_timestamps_start_trial

def Trim_TTLs(trials, DIO_timestamps_start_trial, min_distances):
    if len(trials)!=len(DIO_timestamps_start_trial):
        print(f"unequal numbers of trials between bpod ({len(trials)}) and DIO ({len(DIO_timestamps_start_trial)})")
        if np.argmax(min_distances) == len(trials):
            print("One extra TTL pulse received on DIO at the end of the session")
            DIO_timestamps_start_trial = DIO_timestamps_start_trial[:-1]
            print("extra TTL pulse removed")
    else:
        print("Same number of trials between bpod and DIO")
    return DIO_timestamps_start_trial


def plot_probe(raw_rec, channel_labels):
    y_lim_widget = pn.widgets.EditableRangeSlider(
        name='y_lim', start=0, end=raw_rec.get_channel_locations().max(),
        value=(raw_rec.get_channel_locations().max() - 800, raw_rec.get_channel_locations().max() - 200),
        step=10)

    channels_colors = [channel_label_color_dict[label] for label in channel_labels]

    @pn.depends(y_lim_widget)
    def inspect_probes_channels_labels(ylim):
        fig, axs = plt.subplots(1, 3, figsize=(10, 6))

        sw.plot_probe_map(raw_rec, color_channels=channels_colors, ax=axs[0], with_channel_ids=False)

        sw.plot_probe_map(raw_rec, color_channels=channels_colors, ax=axs[1], with_channel_ids=False)

        patches = [mpatches.Patch(color=color, label=label) for label, color in channel_label_color_dict.items()]

        axs[2].legend(handles=patches, loc='upper left', frameon=False);
        axs[2].axis("off");
        # axs[0].

        axs[1].set_ylim(ylim[0], ylim[1])

        # Draw a rectangle on axs[0] with these ylims
        # Assuming arbitrary x values, here 0 to 10 for illustration
        rect = mpatches.Rectangle((-100, ylim[0]), 600, ylim[1] - ylim[0], linewidth=1, edgecolor='r', facecolor='none')

        axs[0].add_patch(rect)
        plt.close()
        return fig

    return pn.Column(y_lim_widget, pn.pane.Matplotlib(inspect_probes_channels_labels))

def add_custom_metrics_to_phy_folder(raw_rec, path_recording_folder):

    split_preprocessed_recording = raw_rec.split_by("group")

    for group, sub_rec in split_preprocessed_recording.items():
        write_binary_recording(sub_rec,
                               file_paths=f"{path_recording_folder}/spike_interface_output/probe{group}/sorter_output/recording.dat",
                               **job_kwargs)

        params_path = Path(f"{path_recording_folder}/spike_interface_output/probe{group}/sorter_output/params.py")

        # modify params.py to point at .dat file extracted
        with open(params_path, 'r') as file:
            lines = file.readlines()

        with open(params_path, 'w') as file:
            file.writelines(
                ['dat_path = r\'recording.dat\'\n' if line.startswith('dat_path =') else line for line in lines])

        sorting = read_sorter_folder(f"{path_recording_folder}/spike_interface_output/probe{group}")
        # sorting = read_phy(f"{path_recording_folder}/spike_interface_output/probe{group}/sorter_output/")

        # compute 'isi_violation', 'presence_ratio' to add to phy
        analyzer = create_sorting_analyzer(sorting, sub_rec, sparse=True, format="memory", **job_kwargs)

        analyzer.compute({"random_spikes": dict(method="uniform", max_spikes_per_unit=500),
                          "templates": dict(),
                          "noise_levels": dict(),
                          "quality_metrics": dict(metric_names=['isi_violation', 'presence_ratio'])})
        metrics = analyzer.get_extension('quality_metrics').get_data()
        metrics.index.name = "cluster_id"
        metrics.reset_index(inplace=True)
        # create .tsv files in sorter_output folder
        for metric in ['isi_violations_ratio', 'presence_ratio']:
            metrics[["cluster_id", metric]].to_csv(
                f"{path_recording_folder}/spike_interface_output/probe{group}/sorter_output/cluster_{metric}.tsv",
                sep="\t", index=False)

        # Use ks labels as default
        kslabels = pd.read_csv(
            f"{path_recording_folder}/spike_interface_output/probe{group}/sorter_output/cluster_KSLabel.tsv", sep="\t")
        kslabels.rename(columns={"KSLabel": "group"}, inplace=True)
        kslabels.to_csv(f"{path_recording_folder}/spike_interface_output/probe{group}/sorter_output/cluster_group.tsv",
                        sep="\t", index=False)

def extract_protocol(protocol):
    part = protocol.split('_')[1].split('.')[0]
    return ' '.join(re.findall('[A-Z][^A-Z]*', part))

def load_rec_file_with_correct_times(path_recording_folder):

    path_recording, rec_file_name = find_rec_file(path_recording_folder)
    timestamps = get_timestamps_from_rec(path_recording_folder)

    raw_rec = read_spikegadgets(path_recording, use_names_as_ids=False)
    raw_rec.set_times(timestamps)
    print("Correct times assigned")
    return raw_rec


def copy_file_folder(source, target):
    # Ensure the target directory exists
    try:
        os.makedirs(target, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory: {e}")

    # Determine if the source is a file or directory
    if source.is_dir():
        # If source is a directory, use shutil.copytree
        if not os.path.exists(target / source.name):
            print(f"[white on cyan]Copying directory: {source} to {target}[/white on cyan]")
            try:
                shutil.copytree(source, target / source.name)
                print(f"[white on green]Copied directory: {source} to {target}[/white on green]")
            except Exception as e:
                print(f"Error copying directory: {e}")
        else:
            print(f"Directory {target / source.name} already on local SSD")
    elif source.is_file():
        # If source is a file, use shutil.copy
        if not os.path.exists(target / source.name):
            print(f"[white on cyan]Copying file: {source} to {target}[/white on cyan]")
            try:
                shutil.copy(source, target)
                print(f"[white on green]Copied file: {source} to {target}[/white on green]")
            except Exception as e:
                print(f"Error copying file: {e}")
        else:
            print(f"File {target / source.name} already on local SSD")

    return target  # Return the local path to the file or directory

def move_file_folder(source, target):

    if not os.path.exists(target):
        os.makedirs(target)
    else:
        print(f"Directory '{target}' already exists.")

    print(f"[white on cyan]Moving folder:{source} to {target}[/white on cyan]")
    shutil.copytree(source, target, dirs_exist_ok=True)
    print(f"[white on green]Folder moved:{source} to {target}[/white on green]")
    time.sleep(5)
    shutil.rmtree(source)

def get_session_params(bpod_file, path_recording_folder):
    bpod_data = loadmat([q for q in bpod_file if "DetectionConfidence" in str(q)][-1], simplify_cells=True)[
        'SessionData']
    session_params = pd.concat([pd.DataFrame([bpod_data["Custom"]["SessionMeta"]]),
               pd.DataFrame([bpod_data["Custom"]["General"]])], axis=1)
    session_params.to_csv(Path(f"{path_recording_folder}/session_infos.csv"))
    return session_params

def optimize_trials_dtypes(trials):
    for col in trials.columns:
        if trials[col].isin([1, 0, np.nan]).all():
            trials[col] = trials[col].astype('boolean')
    trials["trial_type"]= pd.Categorical(trials["trial_type"],
        categories=['hit', 'miss', 'fa', 'cr'],  # Specify the order of categories
        ordered=True  # Ensure the categories have a defined order
        )
    return trials

def fix_params_file(phy_folder):
    ''' params.py file needs a recording_dat path with .dat extension even if file does not exist'''
    with open(phy_folder / "params.py", 'r') as file:
        lines = file.readlines()
    with open(phy_folder / "params.py", 'w') as file:
        file.writelines(
            ['dat_path = r\'recording.dat\'\n' if line.startswith('dat_path =') else line for line in lines])

def find_bpod_files_and_move_to_SSD( raw_rec, path_recording_folder, SSD_drive, server_folder):
    '''find files on server, and move them to local SSD'''
    bpod_files = find_session_mat_files(
    Path(server_folder, Path(*path_recording_folder.relative_to(f'{path_recording_folder.drive}/').parts[:2]),
         "bpod_session"), path_recording_folder, raw_rec)
    if len( bpod_files)>0:
        for file in bpod_files:
            copy_file_folder(Path(*file.parts[:-1]), SSD_drive / Path(*path_recording_folder.relative_to(f'{path_recording_folder.drive}/').parts[:2]) / "bpod_session")
    else:
        print("no bpod files on server")


def plot_drift(sub_rec, ax):
    # takes 5 min per probe
    noise_levels_int16 = get_noise_levels(sub_rec, return_scaled=False)

    peaks = detect_peaks(sub_rec, method='locally_exclusive', noise_levels=noise_levels_int16,
                         detect_threshold=5, radius_um=50.,
                         **job_kwargs)  # it takes 1 min for one probe job_kwargs = dict(n_jobs=10, progress_bar=True)#total_memory=f"{RAM_GB_to_use}G"
    peak_locations = localize_peaks(sub_rec, peaks, method='center_of_mass', radius_um=50., **job_kwargs)

    ax.scatter(peaks['sample_index'] / sub_rec.sampling_frequency, peak_locations['y'], color='k', marker='.',
               alpha=0.002)
    ax.set_ylim((0, 5000))
    return ax

def check_sorting_curated(path_recording_folder):
    directory = path_recording_folder / "spike_interface_output"

    probe_folders = [folder for folder in os.listdir(directory) if
                     folder.startswith('probe') and os.path.isdir(os.path.join(directory, folder))]

    for probe in probe_folders:
        spike_templates = np.load(
            path_recording_folder / f"spike_interface_output\{probe}\sorter_output\spike_templates.npy")
        spike_clusters = np.load(
            path_recording_folder / f"spike_interface_output\{probe}\sorter_output\spike_clusters.npy")
        print(f"{probe} already curated: {not np.array_equal(spike_templates, spike_clusters)}")

def is_sorting_analyzer_updated(path_recording_folder, sorting_analyzer_path):
    '''Checks wheter merged/split units are included'''
    sorting_analyzer = load_sorting_analyzer( sorting_analyzer_path)
    updated = np.isin(sorting_analyzer.unit_ids, sorting_analyzer.unit_ids).all()
    print(f"{sorting_analyzer_path} is updated: {updated}")
    return updated


def get_trials_colormap(trials):
    frequencies = trials["frequency"].unique()

    cmap = cc.cm.glasbey

    colormap_trials = {("frequency", (freq,)): cmap(i / (len(frequencies) - 1)) for i, freq in enumerate(frequencies)}

    colormap_trials.update({freq: cmap(i / (len(frequencies) - 1)) for i, freq in enumerate(frequencies)})
    colormap_trials.update({
        ("embed_signal", (True,)): "#F79F79",
        ("embed_signal", (False,)): "#A4508B",
        ("made_choice", (True,)): "#BADEFC",
        ("made_choice", (False,)): "#2C2C34",
        ("choice", ("Left",)): "#EB5E55",
        ("choice", ("Right",)): "#3A3335",
        ("broke_fixation", (True,)): "#5FAD56",
        ("broke_fixation", (False,)): "#F2C14E",
        ("early_withdrawal\n&\nbroke_fixation", (True,)): "#D81E5B",
        ("early_withdrawal\n&\nbroke_fixation", (False,)): "#94DAC6",
        ("early_withdrawal", (True,)): "#D81E5B",
        ("early_withdrawal", (False,)): "#94DAC6",
        ("skipped_reward", (True,)): "#7D82B8",
        ("skipped_reward", (False,)): "#F7A9A8",
        ("error or catch", (True,)): "#8F3985",
        ("error or catch", (False,)): "#4BA3C3",
        ("trial_type", ("hit",)): "#1B998B",
        ("trial_type", ("miss",)): "#F5A65B",
        ("trial_type", ("fa",)): "#A90202",
        ("trial_type", ("cr",)): "#1B4079",
        "hit": "#1B998B",
        "miss": "#F5A65B",
        "fa": "#A90202",
        "cr": "#1B4079"
    })

    from matplotlib import cm
    categories = trials["binned_signal_volume"].unique()
    categories_no_nan = np.array([cat for cat in categories if cat == cat])  # Remove NaN
    categories_sorted = np.sort(categories_no_nan)

    # Assign colors using the 'viridis' colormap
    cmap = cm.get_cmap('viridis', len(categories_sorted))  # Get the colormap
    colors = cmap(np.linspace(0, 1, len(categories_sorted)))  # Assign colors to each category

    category_color_map = {('binned_signal_volume', (cat,)): color for cat, color in zip(categories_sorted, colors)}
    category_color_map[('binned_signal_volume', (np.nan,))] = 'grey'

    colormap_trials.update(category_color_map)

    return colormap_trials


def compute_metrics(path_recording_folder, raw_rec, name="sorting_analyzer_post_curation"):
    sorting_analyzer_path = f"{path_recording_folder}/spike_interface_output/sorting_analyzer_post_curation"
    if os.path.exists(sorting_analyzer_path) and not is_sorting_analyzer_updated(path_recording_folder, sorting_analyzer_path) or (not os.path.exists(sorting_analyzer_path)):
        sortings = []
        for probe_n in tqdm(range(len(raw_rec.get_probes()))):
            sortings.append(read_phy(Path(f"{path_recording_folder}/spike_interface_output/probe{probe_n}/sorter_output")))

        sorting = aggregate_units(sortings)

        analyzer = create_sorting_analyzer(sorting, raw_rec, sparse=True, format="binary_folder",
                                           folder=f"{path_recording_folder}/spike_interface_output/sorting_analyzer_post_curation",
                                           overwrite=True, **job_kwargs)

        analyzer.compute({"random_spikes": dict(),
                          "waveforms": dict(),
                          "templates": dict(),
                          "noise_levels": dict(),
                          "quality_metrics": dict(
                              metric_names=['firing_rate', 'presence_ratio', 'snr', 'isi_violation', 'amplitude_cutoff']),
                          "template_metrics": dict(include_multi_channel_metrics=True),
                          "unit_locations": dict(method="monopolar_triangulation")})

        metrics = pd.concat([analyzer.get_extension('quality_metrics').get_data(),
                             analyzer.get_extension("template_metrics").get_data(),
                             pd.DataFrame(analyzer.get_extension('unit_locations').get_data(),
                                          index=analyzer.unit_ids,
                                          columns=["x (µm)", "y (µm)", "z (µm)"])], axis=1)  #

        metrics.index.name = "unit_id"
        metrics.to_csv(Path(f"{path_recording_folder}/spike_interface_output/metrics_post_curation.csv"))

        is_sorting_analyzer_updated(path_recording_folder,
                                    f"{path_recording_folder}/spike_interface_output/sorting_analyzer_post_curation")



###### higher levels here

def load_trials(path_recording_folder):
    '''Load trials and recmpotes them if not updated'''
    trials = pd.read_csv(Path(f"{path_recording_folder}/trials.csv"))
    trials = optimize_trials_dtypes(trials)

    if ("processing_version" in trials.columns) and (trials["processing_version"].unique() == trial_extract_version):
        print("Trials.csv is updated")
        return trials
    else:
        print("Trials.csv is not updated. \nRecomputing it now.")
        get_trials(path_recording_folder)
        trials = pd.read_csv(Path(f"{path_recording_folder}/trials.csv"))
        trials = optimize_trials_dtypes(trials)
        return trials

def get_trials(path_recording_folder):
    '''Process trials and extract analogIO and DIO'''

    path_recording_folder = Path(path_recording_folder)
    raw_rec = load_rec_file_with_correct_times(path_recording_folder)

    print(
        f"Recording duration in minutes: {raw_rec.get_total_duration() / 60}, sampling rate: {raw_rec.get_sampling_frequency()} Hz")
    print(f"Probes present: {raw_rec.get_probes()}")

    gaps_start_stop = check_timestamps_gaps(raw_rec.get_times())
    call_trodesexport(path_recording_folder, "dio")
    call_trodesexport(path_recording_folder, "analogio")

    DIO_timestamps_start_trial = select_DIO_sync_trial_trace(path_recording_folder)

    bpod_file = find_session_mat_files(path_recording_folder.parent.parent / "bpod_session", path_recording_folder,
                                       raw_rec)
    get_session_params(bpod_file, path_recording_folder)

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    trials = extract_trials(bpod_file, DIO_timestamps_start_trial, axs=axs)
    clean_trials(trials, gaps_start_stop, path_recording_folder)
    plt.savefig(path_recording_folder / 'trials_concatenation.png')  # Save to a file



def spikesort(path_recording_folder):
    """Spike sort session and extract trials"""

    path_recording_folder = Path(path_recording_folder)


    if "ottlabfs" in path_recording_folder.drive:
        rec_file_on_server = True
        if not os.path.exists(Path("X:/") / path_recording_folder.relative_to("//ottlabfs.bccn-berlin.pri/ottlab") /
                              path_recording_folder.parts[-1]):
            path_recording_folder = copy_file_folder(path_recording_folder / path_recording_folder.parts[-1],
                                                     Path("X:/") / path_recording_folder.relative_to(
                                                         "//ottlabfs.bccn-berlin.pri/ottlab"))
        else:
            path_recording_folder = Path("X:/") / path_recording_folder.relative_to("//ottlabfs.bccn-berlin.pri/ottlab")
    else:
        rec_file_on_server = False


    raw_rec = load_rec_file_with_correct_times(path_recording_folder)

    #copy bpod file from server if necessary

    bpod_files = find_session_mat_files(
        Path(server_folder, Path(*path_recording_folder.relative_to(f'{path_recording_folder.drive}/').parts[:2]),
             "bpod_session"), path_recording_folder, raw_rec)
    for file in bpod_files:
        copy_file_folder(Path(*file.parts[:-1]), Path(*path_recording_folder.parts[:3], "bpod_session"))


    print(f"Recording duration in minutes: {raw_rec.get_total_duration()/60}, sampling rate: {raw_rec.get_sampling_frequency()} Hz")
    print(f"Probes present: {raw_rec.get_probes()}")

    get_trials(path_recording_folder)

    if os.path.exists(f"{path_recording_folder}/channel_labels.csv"):
        print("channel_label.csv already exists!")
        channel_labels_df = pd.read_csv(f"{path_recording_folder}/channel_labels.csv")
    else:
        channel_labels_list = []
        channel_ids_list = []
        # detect noisy, dead, and out-of-brain channels
        split_preprocessed_recording = raw_rec.split_by("group")
        for group, sub_rec in tqdm(split_preprocessed_recording.items()):
            bad_channel_ids, channel_labels = detect_bad_channels(sub_rec, method='coherence+psd')

            channel_labels_list.extend(channel_labels)
            channel_ids_list.extend(sub_rec.channel_ids)

        channel_labels_df = pd.DataFrame([channel_ids_list, channel_labels_list],
                                         index=["channel_ids", "channel_labels"]).T
        os.makedirs(Path(f"{path_recording_folder}"), exist_ok=True)
        channel_labels_df.to_csv(Path(f"{path_recording_folder}/channel_labels.csv"), index=False)

    bad_channel_ids = channel_labels_df[~(channel_labels_df["channel_labels"] == "good")]["channel_ids"].values.astype(
        str)

    print(channel_labels_df["channel_labels"].value_counts(), "\n \n",
          channel_labels_df[channel_labels_df["channel_labels"] != "good"])

    # remove bad channels
    raw_rec = raw_rec.remove_channels(bad_channel_ids)
    print(f"{len(bad_channel_ids)} bad channels removed")


    ##### use spikeinterface
    # split_preprocessed_recording = raw_rec.split_by("group")
    # for group, sub_rec in tqdm(split_preprocessed_recording.items()):
    #     torch.cuda.empty_cache()
    #     sorting = run_sorter(sorter_name="kilosort4",
    #                          recording=sub_rec,
    #                          folder=f"{path_recording_folder}/spike_interface_output/probe{group}",
    #                          verbose=False,
    #                          remove_existing_folder=True,
    #                          torch_device="cuda",
    #                          use_binary_file = True
    #     )

    ##### use directly kilosort

    split_preprocessed_recording = raw_rec.split_by("group")
    for probe_n, sub_rec in split_preprocessed_recording.items():
        binary_file_path = f"{path_recording_folder}/spike_interface_output/probe{probe_n}/sorter_output/recording.dat"
        probe_filename = f"{path_recording_folder}/spike_interface_output/probe{probe_n}/sorter_output/probe.prb"
        if not os.path.exists(binary_file_path):
            os.makedirs(Path(binary_file_path).parent, exist_ok=True)
            write_binary_recording(
                recording=sub_rec,
                file_paths=binary_file_path, **job_kwargs)

            pg = sub_rec.get_probegroup()
            write_prb(probe_filename, pg)
            print(f"probe {probe_n} completed \n")

        probe = load_probe(probe_filename)
        settings = {'filename': binary_file_path,
                    "n_chan_bin": probe["n_chan"], "fs": sub_rec.get_sampling_frequency()}#try nskip 20

        result_dir = path_recording_folder / "spike_interface_output" / f"probe{probe_n}" / "sorter_output"
        os.makedirs(result_dir, exist_ok=True)

        run_kilosort(settings=settings, probe=probe, data_dtype=sub_rec.get_dtype(),
                     device=torch.device("cuda"), results_dir=result_dir, clear_cache=True)

        Path(binary_file_path).unlink()


    #####compute metrics
    for group, sub_rec in split_preprocessed_recording.items():
        # sorting = read_sorter_folder(f"{path_recording_folder}/spike_interface_output/probe{probe_n}")
        sorting = read_kilosort(f"{path_recording_folder}/spike_interface_output/probe{probe_n}/sorter_output")

        analyzer = create_sorting_analyzer(sorting, sub_rec, sparse=True, format="binary_folder",
                                           folder=f"{path_recording_folder}/spike_interface_output/probe{group}/sorting_analyzer",
                                           overwrite=True, **job_kwargs)

        analyzer.compute({"random_spikes": dict(),
                          "waveforms": dict(),
                          "templates": dict(),
                          "noise_levels": dict(),
                          "quality_metrics": dict(metric_names=['firing_rate', 'presence_ratio', 'snr', 'isi_violation',
                                                                'amplitude_cutoff'])})

        metrics = analyzer.get_extension('quality_metrics').get_data()
        metrics.index.name = "cluster_id"

        for column_name in metrics.columns:
            if column_name not in ["num_spikes", "firing_rate"]:  # already computed by phy
                metric = metrics[column_name].reset_index()
                metric.to_csv(
                    f"{path_recording_folder}/spike_interface_output/probe{group}/sorter_output/cluster_{column_name}.tsv",
                    sep="\t", index=False)

        metrics["probe_n"] = f"probe{group}"
        metrics.index.name = "unit_id"
        metrics.to_csv(Path(f"{path_recording_folder}/spike_interface_output/probe{group}/metrics.csv"))

    # if os.path.exists(path_recording_folder / "binary"):
    #     print("remove temp  binaries created with spikeinterface")
    #     shutil.rmtree (path_recording_folder / "binary")

    # move to server
    # move_file_folder(path_recording_folder,
    #                  Path(server_folder, path_recording_folder.relative_to(f'{path_recording_folder.drive}/')))


    variables_to_delete = ["sorting", "sub_rec", "raw_rec", "split_preprocessed_recording", "analyzer", "metrics"]

    for var in variables_to_delete:
        if var in locals():
            del locals()[var]

    #move to D
    big_HD = r"D:\Roberto\processed"
    move_file_folder(path_recording_folder,
                     Path(big_HD, path_recording_folder.relative_to(f'{path_recording_folder.drive}/')))

  # remove .rec file

  #
  #   if rec_file_on_server:
  #       Path.unlink(
  #           path_recording_folder / path_recording_folder.parts[-1])  # delete rec file locally if already on server
  #
