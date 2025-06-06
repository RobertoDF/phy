
from phy import IPlugin
from phy.cluster.views import ManualClusteringView  # Base class for phy views
from phy.plot.plot import PlotCanvasMpl  # matplotlib canvas
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from tqdm import (tqdm)
from spikeinterface.extractors import read_phy
import re
from spikeinterface import load_sorting_analyzer

class EventView(ManualClusteringView):
    plot_canvas_class = PlotCanvasMpl  # use matplotlib instead of OpenGL (the default)

    def __init__(self, c=None, trials=None, spike_times=None, **kwargs):
        """features is a function (cluster_id => Bunch(data, ...)) where data is a 3D array."""
        super(EventView, self).__init__()
        self.controller = c
        self.model = c.model
	columns = ["noise_start_time", "stimulus_start_time", "choice_time", "reward_start_time"]

	# Check for missing columns
	missing_columns = [col for col in columns if col not in trials.columns]
	if missing_columns:
    		print(f"Warning: The trials DataFrame is missing the following columns: {missing_columns}")

        self.confidence_start_times = trials.query("stimulus_name == 'Detection Confidence'")[
            columns].reset_index(drop=True)
        self.aud_stim_start = trials.query("stimulus_name == 'Auditory Tuning'")["stimulus_start_time"].rename("pure_tone_start_time")
        self.spike_times = spike_times
        self.window = (1, 2)# window psth
        self.binsize = 0.01

    def on_request_similar_clusters(self, cid=None):
        self.on_select()

    def on_select(self, cluster_ids=(), **kwargs):
        self.cluster_ids = cluster_ids
        # We don't display anything if no clusters are selected.
        if not cluster_ids:
            return

        self.canvas.ax.clear()

        self.canvas.subplots(len(cluster_ids)) #len(cluster_ids)

        for i, d in enumerate(cluster_ids):
            sp = self.spike_times[d]

            df = pd.concat([self.calc_hists(sp, self.aud_stim_start, self.window,  self.binsize),
                   self.calc_hists(sp, self.confidence_start_times["noise_start_time"], self.window,  self.binsize),
                   self.calc_hists(sp, self.confidence_start_times["stimulus_start_time"],  self.window,  self.binsize),
                   self.calc_hists(sp, self.confidence_start_times["choice_time"],  self.window,  self.binsize),
                   self.calc_hists(sp, self.confidence_start_times["reward_start_time"],  self.window,  self.binsize)
                   ], axis=1)
            df.index.name = "Time (s)"
            df.plot(ax=self.canvas.axes[i][0], cmap="Set1", alpha=.5)
            self.canvas.axes[i][0].set_ylabel = f"Spikes/{self.binsize} s"

            if i==0:
                self.canvas.axes[i][0].legend()
            sns.despine(ax=self.canvas.axes[i][0])
        self.canvas.update()

        return

    def calc_hists(self, spike_times, stim_start, window, binsize):
        if stim_start.shape[0]>0:
            start = np.searchsorted(spike_times, stim_start - window[0])
            end = np.searchsorted(spike_times, stim_start + window[1])
            count, bins = np.histogram(
                np.concatenate([spike_times[start[i]:end[i]] - stim_start[i] for i in range(len(start))]),
                bins=np.arange(-window[0], window[1], binsize))
            return pd.Series(count, name=stim_start.name, index=bins[:-1])/len(stim_start)
        else:
            return pd.Series()
            
current_dir = Path(os.getcwd())
# Get the parent directory two levels up
spike_folder = Path(*current_dir.parts[:-2])
path_recording_folder = Path(*current_dir.parts[:-3])

if spike_folder:
    # Extract the suffix if present (e.g., "_0" from "spike_interface_output_0")
    suffix = spike_folder.name.replace("spike_interface_output", "")
    # Construct the trials filename with the same suffix
    trials_filename = f"trials{suffix}.csv"
    trials_path = Path(*current_dir.parts[:-3]) / trials_filename
else:
    # Fall back to default if no matching folder found
    trials_path = Path(*current_dir.parts[:-3]) / "trials.csv"

print(trials_path)
trials = pd.read_csv(trials_path)


# Parses last fields parameter (<time uint32><...>) as a single string
# Assumes it is formatted as <name number * type> or <name type>
# Returns: np.dtype
def parseFields(fieldstr):
    # Returns np.dtype from field string
    sep = re.split('\s', re.sub(r"\>\<|\>|\<", ' ', fieldstr).strip())
    # print(sep)
    typearr = []
    # Every two elmts is fieldname followed by datatype
    for i in range(0, sep.__len__(), 2):
        fieldname = sep[i]
        repeats = 1
        ftype = 'uint32'
        # Finds if a <num>* is included in datatype
        if sep[i + 1].__contains__('*'):
            temptypes = re.split('\*', sep[i + 1])
            # Results in the correct assignment, whether str is num*dtype or dtype*num
            ftype = temptypes[temptypes[0].isdigit()]
            repeats = int(temptypes[temptypes[1].isdigit()])
        else:
            ftype = sep[i + 1]
        try:
            fieldtype = getattr(np, ftype)
        except AttributeError:
            print(ftype + " is not a valid field type.\n")
            exit(1)
        else:
            typearr.append((str(fieldname), fieldtype, repeats))
    return np.dtype(typearr)

def readTrodesExtractedDataFile(filename):
    with open(filename, 'rb') as f:
        # Check if first line is start of settings block
        if f.readline().decode('ascii').strip() != '<Start settings>':
            raise Exception("Settings format not supported")
        fields = True
        fieldsText = {}
        for line in f:
            # Read through block of settings
            if (fields):
                line = line.decode('ascii').strip()
                # filling in fields dict
                if line != '<End settings>':
                    vals = line.split(': ')
                    fieldsText.update({vals[0].lower(): vals[1]})
                # End of settings block, signal end of fields
                else:
                    fields = False
                    dt = parseFields(fieldsText['fields'])
                    fieldsText['data'] = np.zeros([1], dtype=dt)
                    break
        # Reads rest of file at once, using dtype format generated by parseFields()
        dt = parseFields(fieldsText['fields'])
        data = np.fromfile(f, dt)
        fieldsText.update({'data': data})
        return fieldsText

folder_name = spike_folder.name  # "spike_interface_output_0"
if "_" in folder_name:
    base, suffix = folder_name.rsplit("_", 1)  # Split on last '_'
    has_suffix = suffix.isdigit()  # True if suffix is numeric (e.g., "0", "1")
else:
    has_suffix = False


if not has_suffix:
    try:
        timestamps_dict = readTrodesExtractedDataFile(Path(*current_dir.parts[:-3])/f"{current_dir.parts[-4].split('.rec')[0]}.analog" /f"{current_dir.parts[-4].split('.rec')[0]}.timestamps.dat")
    except FileNotFoundError:
        print(".time Folder missing")

    timestamps = timestamps_dict["data"]["time"]
    times = timestamps/float(timestamps_dict["clockrate"])

elif has_suffix:
    probe_n = int(re.findall(r"\d+", Path(*current_dir.parts[:-1]).name)[-1])
    probe_str = "np2"
    clock_str = [f'{probe_str}-{probe_id}-clock_{suffix}.raw' for probe_id in ["a", "b"]][probe_n]


    dt = {'names': ('time', 'acq_clk_hz', 'block_read_sz', 'block_write_sz'),
      'formats': ('datetime64[us]', 'u4', 'u4', 'u4')}
    meta = np.genfromtxt(os.path.join(path_recording_folder, f'start-time_{suffix}.csv'), delimiter=',', dtype=dt, skip_header=1)

    df = pd.read_csv(os.path.join(path_recording_folder, f'start-time_{suffix}.csv')) # we need this because times with ofsets are not parsed properly in numpy
    df['Timestamp'] = pd.to_datetime(df['Timestamp'].str.replace('\+02:00$', '', regex=True),
                                 format='%Y-%m-%dT%H:%M:%S.%f')

    meta["time"] = df['Timestamp'].values

    print(os.path.join(path_recording_folder, clock_str))
    times = np.fromfile(os.path.join(path_recording_folder, clock_str), dtype=np.uint64).astype(np.double) / meta['acq_clk_hz']

sorting_path = Path(*current_dir.parts[:-1]) / "sorter_output"
sorting = read_phy(sorting_path)# This should work even if phy was not used

spike_times = {}
print("Extract spike times")
for unit_id in tqdm(sorting.unit_ids):
    spike_train = sorting.get_unit_spike_train(unit_id=unit_id)
    valid_indices = spike_train[spike_train < len(times)]  #sometimes out of bond spiketimes
    spike_times[unit_id] = times[valid_indices]

class EventViewPlugin(IPlugin):
    def attach_to_controller(self, controller):
        def create_event_view():
            """A function that creates and returns a view."""
            try:
                return EventView(c=controller, trials=trials, spike_times=spike_times)
            except:
                print("Can´t add EventViewPlugin, chill out phy still works :)")

        controller.view_creator['EventView'] = create_event_view
