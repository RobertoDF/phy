"""Plot trial-aligned spike counts using recording clocks and live Phy clusters."""

import csv
import logging
import re
from pathlib import Path

import matplotlib as mpl
import numpy as np

from phy import IPlugin
from phy.cluster.views import ManualClusteringView
from phy.plot.plot import PlotCanvasMpl

logger = logging.getLogger(__name__)

EVENT_COLUMNS = ('noise_start_time', 'stimulus_start_time', 'choice_time', 'reward_start_time')
EVENT_NAMES = ('pure_tone_start_time',) + EVENT_COLUMNS


def _data_paths(current_dir):
    """Resolve recording files from <output>/probeN/<Phy export folder>."""
    current_dir = Path(current_dir)
    output = current_dir.parents[1]
    recording = current_dir.parents[2]
    match = re.fullmatch(r'spike_interface_output(_\d+)?', output.name)
    probe = re.fullmatch(r'probe(\d+)', current_dir.parent.name)
    if match is None or probe is None:
        raise ValueError(f'Expected spike_interface_output[_N]/probeN/<export> at {current_dir}')
    suffix = match.group(1) or ''
    trials = recording / f'trials{suffix}.csv'
    if suffix:
        probe_index = int(probe.group(1))
        if probe_index not in (0, 1):
            raise ValueError(f'No np2 clock mapping for {current_dir.parent}')
        clock = recording / f'np2-{"ab"[probe_index]}-clock{suffix}.raw'
        return trials, clock, recording / f'start-time{suffix}.csv'
    stem = recording.name.removesuffix('.rec')
    return trials, recording / f'{stem}.analog' / f'{stem}.timestamps.dat', None


def _load_events(path):
    events = {name: [] for name in EVENT_NAMES}
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        missing = set(('stimulus_name',) + EVENT_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f'{path}: missing trial columns {sorted(missing)}')
        for row_number, row in enumerate(reader, start=2):
            if row['stimulus_name'] == 'Auditory Tuning':
                columns = [('pure_tone_start_time', 'stimulus_start_time')]
            elif row['stimulus_name'] == 'Detection Confidence':
                columns = [(name, name) for name in EVENT_COLUMNS]
            else:
                continue
            for name, column in columns:
                value = row[column]
                try:
                    event = float(value) if value and value.strip() else np.nan
                except ValueError as error:
                    raise ValueError(f'{path}: row {row_number}, {column}: {value!r}') from error
                events[name].append(event)
    for name, values in events.items():
        values = np.asarray(values, dtype=float)
        finite = np.isfinite(values)
        if not finite.all():
            logger.warning(
                '%s: ignoring %d missing/non-finite %s events.', path, int((~finite).sum()), name
            )
        events[name] = values[finite]
    return events


def _trodes_dtype(fields):
    tokens = re.findall(r'<([^<>]+)>', fields)
    if not tokens or ''.join(f'<{token}>' for token in tokens) != fields.strip():
        raise ValueError(f'Invalid Trodes fields: {fields!r}')
    dtype = []
    for token in tokens:
        name, kind = token.split()
        repeat = 1
        if '*' in kind:
            left, right = kind.split('*')
            repeat, kind = (int(left), right) if left.isdigit() else (int(right), left)
        if (
            kind
            not in (
                'uint8',
                'uint16',
                'uint32',
                'uint64',
                'int8',
                'int16',
                'int32',
                'int64',
                'float32',
                'float64',
            )
            or repeat < 1
        ):
            raise ValueError(f'Invalid Trodes field type: {token!r}')
        dtype.append((name, kind) if repeat == 1 else (name, kind, (repeat,)))
    return np.dtype(dtype)


def _load_clock(path, metadata):
    """Memory-map integer timestamps; convert only selected spikes to seconds."""
    if metadata is not None:
        with metadata.open(newline='', encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 1:
            raise ValueError(f'{metadata}: expected one acquisition clock metadata row')
        rate = float(rows[0]['acq_clk_hz'])
        dtype, offset = np.dtype('uint64'), 0
    else:
        settings = {}
        with path.open('rb') as stream:
            if stream.readline().strip() != b'<Start settings>':
                raise ValueError(f'{path}: missing Trodes <Start settings>')
            while True:
                line = stream.readline()
                if not line or stream.tell() > 65536:
                    raise ValueError(f'{path}: missing Trodes <End settings>')
                if line.strip() == b'<End settings>':
                    break
                key, value = line.decode('ascii').strip().split(':', 1)
                settings[key.lower()] = value.strip()
            offset = stream.tell()
        rate = float(settings['clockrate'])
        dtype = _trodes_dtype(settings['fields'])
        if 'time' not in dtype.names or dtype['time'] != np.dtype('uint32'):
            raise ValueError(f'{path}: expected a scalar uint32 Trodes time field')
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError(f'{metadata or path}: clock rate must be finite and positive')
    size = path.stat().st_size - offset
    if size <= 0 or size % dtype.itemsize:
        raise ValueError(f'{path}: empty or truncated timestamp data')
    timestamps = np.memmap(path, dtype=dtype, mode='r', offset=offset)
    if metadata is None:
        timestamps = timestamps['time']
    return timestamps, rate


def _cluster_times(controller, cluster_id, timestamps, rate):
    spike_ids = controller.supervisor.clustering.spikes_per_cluster.get(cluster_id, ())
    samples = np.asarray(controller.model.spike_samples[np.asarray(spike_ids, dtype=np.int64)])
    if samples.dtype.kind not in 'iu':
        raise ValueError('Phy spike_samples must be integer sample indices')
    valid = (samples >= 0) & (samples < len(timestamps))
    if not valid.all():
        logger.warning(
            'Cluster %s: ignoring %d spike samples outside the recording clock.',
            cluster_id,
            int((~valid).sum()),
        )
    times = np.asarray(timestamps[samples[valid]], dtype=float) / rate
    # Searchsorted requires chronological order, also after merges or clock resets.
    return np.sort(times)


def _peri_event_histogram(spike_times, events, window=(1, 2), binsize=0.01):
    """Counts per finite trial, retaining the original arange/histogram edge rules."""
    events = np.asarray(events, dtype=float)
    if not np.isfinite(events).all():
        raise ValueError('Event times must be finite')
    bins = np.arange(-window[0], window[1], binsize)
    counts = np.zeros(len(bins) - 1, dtype=np.int64)
    starts = np.searchsorted(spike_times, events - window[0])
    ends = np.searchsorted(spike_times, events + window[1])
    # Bound temporary storage even for overlapping windows and very active units.
    for event, start, end in zip(events, starts, ends):
        for chunk_start in range(start, end, 65536):
            relative = spike_times[chunk_start : min(chunk_start + 65536, end)] - event
            counts += np.histogram(relative, bins=bins)[0]
    return bins[:-1], counts / len(events) if len(events) else np.full(len(counts), np.nan)


class EventView(ManualClusteringView):
    plot_canvas_class = PlotCanvasMpl

    def __init__(self, controller, events, timestamps, rate):
        super().__init__()
        self.controller = controller
        self.events = events
        self.timestamps = timestamps
        self.rate = rate
        self.window = (1, 2)
        self.binsize = 0.01
        self._lines = []
        self._scroll_callbacks = set()

    def on_request_similar_clusters(self, cid=None):
        self.on_select(self.cluster_ids)

    def on_select(self, cluster_ids=(), **kwargs):
        self.cluster_ids = list(cluster_ids)
        nrows = max(1, len(cluster_ids))
        if len(self._lines) != nrows:
            # Retain Phy's zoom behavior without accumulating callbacks for discarded axes.
            canvas = self.canvas.figure.canvas
            for callback in self._scroll_callbacks:
                canvas.mpl_disconnect(callback)
            before = set(canvas.callbacks.callbacks.get('scroll_event', {}))
            self.canvas.subplots(nrows)
            self._scroll_callbacks = (
                set(canvas.callbacks.callbacks.get('scroll_event', {})) - before
            )
            self._lines = []
            colors = mpl.colormaps['Set1'](np.linspace(0, 1, 5))
            for ax in self.canvas.axes[:, 0]:
                self._lines.append(
                    [
                        ax.plot([], [], label=name, color=color, alpha=0.5)[0]
                        for name, color in zip(EVENT_NAMES, colors)
                    ]
                )
                ax.set_xlabel('Time (s)')
                ax.set_ylabel(f'Spikes/{self.binsize} s / trial')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            self.canvas.axes[0, 0].legend()
        for i, ax in enumerate(self.canvas.axes[:, 0]):
            ax.set_title(f'Cluster {cluster_ids[i]}' if cluster_ids else '')
            spikes = (
                _cluster_times(self.controller, cluster_ids[i], self.timestamps, self.rate)
                if cluster_ids
                else np.array([])
            )
            for line, events in zip(self._lines[i], self.events.values()):
                x, y = _peri_event_histogram(spikes, events, self.window, self.binsize)
                line.set_data(x if cluster_ids else [], y if cluster_ids else [])
            ax.relim()
            ax.autoscale_view()
        self.canvas.update()


class EventViewPlugin(IPlugin):
    def attach_to_controller(self, controller):
        def create_event_view():
            current_dir = '<unresolved working directory>'
            try:
                current_dir = Path.cwd()
                trials, clock, metadata = _data_paths(current_dir)
                events = _load_events(trials)
                timestamps, rate = _load_clock(clock, metadata)
                # Validate the controller before allocating a Qt-backed canvas.
                controller.model.spike_samples
                controller.supervisor.clustering.spikes_per_cluster
                return EventView(controller, events, timestamps, rate)
            except Exception:
                # Optional GUI plugin boundary: malformed/missing data must not abort Phy.
                logger.warning('Cannot create EventView from %s.', current_dir, exc_info=True)
                return None

        controller.view_creator['EventView'] = create_event_view
