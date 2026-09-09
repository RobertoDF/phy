"""Headless regressions for the independently installed recording view plugins."""

import csv
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.testing import assert_allclose, assert_array_equal

from phy.cluster.clustering import Clustering
from phy.utils.plugin import IPluginRegistry, discover_plugins

PLUGIN_DIR = Path(
    os.environ.get('PHY_PLUGIN_TEST_DIR', Path(__file__).resolve().parents[3] / 'plugins')
)
PLUGIN_NAMES = ('EventViewPlugin', 'WaveformSpikeinterfaceViewPlugin')


def _import_plugin(name):
    spec = importlib.util.spec_from_file_location(name, PLUGIN_DIR / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def event():
    return _import_plugin('EventViewPlugin')


@pytest.fixture
def waveform():
    return _import_plugin('WaveformSpikeinterfaceViewPlugin')


class _Canvas:
    def __init__(self):
        self.figure = Figure()
        FigureCanvasAgg(self.figure)
        self.axes = self.figure.subplots(squeeze=False)
        self.updates = 0

    @property
    def ax(self):
        return self.axes[0, 0]

    def attach_events(self, view):
        pass

    def subplots(self, nrows=1):
        self.figure.clear()
        self.axes = self.figure.subplots(nrows, 1, squeeze=False)
        for ax in self.axes[:, 0]:
            self.figure.canvas.mpl_connect('scroll_event', lambda event, ax=ax: ax)

    def update(self):
        self.updates += 1


def _controller(samples=(2, 5, 8, 10), assignments=(0, 0, 1, 1), sample_rate=1000):
    return SimpleNamespace(
        view_creator={},
        model=SimpleNamespace(spike_samples=np.asarray(samples), sample_rate=sample_rate),
        supervisor=SimpleNamespace(clustering=Clustering(np.asarray(assignments))),
    )


def _write_trials(path, rows):
    columns = (
        'stimulus_name',
        'noise_start_time',
        'stimulus_start_time',
        'choice_time',
        'reward_start_time',
    )
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


@pytest.mark.parametrize('name', PLUGIN_NAMES)
def test_import_no_path_resolution(monkeypatch, tmp_path, name):
    monkeypatch.chdir(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail('Plugin resolved/read data at import or attachment')

    # Phy/Matplotlib dependencies have already been imported by test collection.
    monkeypatch.setattr(Path, 'cwd', forbidden)
    monkeypatch.setattr(os, 'getcwd', forbidden)
    monkeypatch.setattr(Path, 'open', forbidden)
    module = _import_plugin(name)
    getattr(module, name)().attach_to_controller(_controller())


def test_real_discovery(monkeypatch, tmp_path, caplog):
    plugin_dir = tmp_path / 'plugins'
    plugin_dir.mkdir()
    for name in PLUGIN_NAMES:
        (plugin_dir / f'{name}.py').write_bytes((PLUGIN_DIR / f'{name}.py').read_bytes())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(IPluginRegistry, 'plugins', [])
    names = {plugin.__name__ for plugin in discover_plugins([plugin_dir])}
    assert set(PLUGIN_NAMES) <= names
    assert not [record for record in caplog.records if record.levelname in ('WARNING', 'ERROR')]


@pytest.mark.parametrize(
    'name,view_name',
    [
        ('EventViewPlugin', 'EventView'),
        ('WaveformSpikeinterfaceViewPlugin', 'WaveformSpikeinterfaceView'),
    ],
)
def test_missing_data_skips_before_widget(monkeypatch, tmp_path, caplog, name, view_name):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('PHY_WAVEFORM_ANALYZER', raising=False)
    module = _import_plugin(name)
    monkeypatch.setattr(module, view_name, lambda *a, **kw: pytest.fail('Widget constructed'))
    controller = _controller()
    getattr(module, name)().attach_to_controller(controller)
    assert controller.view_creator[view_name]() is None
    assert f'Cannot create {view_name}' in caplog.text


@pytest.mark.parametrize(
    'suffix,probe,clock_name',
    [
        ('', 1, 'run.timestamps.dat'),
        ('_3', 0, 'np2-a-clock_3.raw'),
        ('_3', 1, 'np2-b-clock_3.raw'),
    ],
)
def test_recording_paths(event, tmp_path, suffix, probe, clock_name):
    root = tmp_path / 'run.rec'
    cwd = root / f'spike_interface_output{suffix}' / f'probe{probe}' / 'sorter_output'
    trials, clock, metadata = event._data_paths(cwd)
    assert trials == root / f'trials{suffix}.csv'
    assert clock.name == clock_name
    assert clock.parent == (root / 'run.analog' if not suffix else root)
    assert metadata == (root / f'start-time{suffix}.csv' if suffix else None)


def test_invalid_path_and_fields(event, tmp_path):
    with pytest.raises(ValueError, match='Expected'):
        event._data_paths(tmp_path / 'wrong' / 'probe1' / 'sorter_output')
    with pytest.raises(ValueError, match='clock mapping'):
        event._data_paths(tmp_path / 'spike_interface_output_1' / 'probe2' / 'sorter_output')
    assert event._trodes_dtype('<time uint32><data 2*int16>')['data'].shape == (2,)
    assert event._trodes_dtype('<time uint32><data int16*2>')['data'].shape == (2,)
    with pytest.raises(ValueError):
        event._trodes_dtype('<time invalid>')


@pytest.mark.parametrize('suffix', ['', '_2'])
def test_load_clock_lazy_creation(event, tmp_path, monkeypatch, suffix):
    root = tmp_path / 'run.rec'
    cwd = root / f'spike_interface_output{suffix}' / 'probe1' / 'sorter_output'
    cwd.mkdir(parents=True)
    trials, clock, metadata = event._data_paths(cwd)
    _write_trials(trials, [['Auditory Tuning', '', 1, '', '']])
    samples = np.arange(32, dtype='uint64' if suffix else 'uint32')
    if suffix:
        metadata.write_text('Timestamp,acq_clk_hz\n2023-12-21T16:13:13+02:00,10\n')
        samples.tofile(clock)
    else:
        clock.parent.mkdir()
        clock.write_bytes(
            b'<Start settings>\nclockrate: 10\nfields: <time uint32>\n'
            b'<End settings>\n' + samples.tobytes()
        )
    timestamps, rate = event._load_clock(clock, metadata)
    assert isinstance(timestamps, np.memmap)
    assert rate == 10
    assert_array_equal(timestamps, samples)
    assert_allclose(event._cluster_times(_controller(), 0, timestamps, rate), [0.2, 0.5])
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(event.EventView, 'plot_canvas_class', _Canvas)
    controller = _controller()
    event.EventViewPlugin().attach_to_controller(controller)
    view = controller.view_creator['EventView']()
    assert isinstance(view, event.EventView)
    view.on_select([0])
    assert view.canvas.ax.get_ylabel() == 'Spikes/0.01 s / trial'


def test_invalid_trials_and_clock_before_widget(event, tmp_path, monkeypatch, caplog):
    cwd = tmp_path / 'run.rec' / 'spike_interface_output_1' / 'probe1' / 'sorter_output'
    cwd.mkdir(parents=True)
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(event, 'EventView', lambda *a: pytest.fail('Widget constructed'))
    controller = _controller()
    event.EventViewPlugin().attach_to_controller(controller)
    trials, clock, metadata = event._data_paths(cwd)
    trials.write_text('wrong\n1\n')
    assert controller.view_creator['EventView']() is None
    assert str(trials) in caplog.text and 'missing trial columns' in caplog.text
    _write_trials(trials, [['Auditory Tuning', '', 'invalid', '', '']])
    assert controller.view_creator['EventView']() is None
    assert 'row 2, stimulus_start_time' in caplog.text
    _write_trials(trials, [['Auditory Tuning', '', '1', '', '']])
    metadata.write_text('acq_clk_hz\n0\n')
    assert controller.view_creator['EventView']() is None
    assert 'clock rate' in caplog.text
    metadata.write_text('acq_clk_hz\n10\n')
    clock.write_bytes(b'\x00')
    assert controller.view_creator['EventView']() is None
    assert 'truncated' in caplog.text


def test_trodes_invalid_headers(event, tmp_path):
    path = tmp_path / 'timestamps.dat'
    for contents in [b'invalid', b'<Start settings>\nfields: <time uint32>\n']:
        path.write_bytes(contents)
        with pytest.raises(ValueError, match='Trodes'):
            event._load_clock(path, None)


def test_trial_index_missing_events(event, tmp_path, caplog):
    trials = tmp_path / 'trials.csv'
    _write_trials(
        trials,
        [
            ['Detection Confidence', 1, 2, 3, 4],
            ['Auditory Tuning', '', 10, '', ''],
            ['Detection Confidence', 'nan', 12, '', 'inf'],
            ['Auditory Tuning', '', 20, '', ''],
        ],
    )
    events = event._load_events(trials)
    assert_array_equal(events['pure_tone_start_time'], [10, 20])
    assert_array_equal(events['noise_start_time'], [1])
    assert_array_equal(events['stimulus_start_time'], [2, 12])
    assert 'ignoring 1' in caplog.text


def _reference_hist(spikes, events, window=(1, 2), binsize=0.01):
    starts = np.searchsorted(spikes, events - window[0])
    ends = np.searchsorted(spikes, events + window[1])
    return np.histogram(
        np.concatenate(
            [spikes[start:end] - event for start, end, event in zip(starts, ends, events)]
        ),
        bins=np.arange(-window[0], window[1], binsize),
    )[0] / len(events)


def test_histogram_reference_edges_and_bounded_work(event, monkeypatch):
    bins = np.arange(-1, 2, 0.01)
    spikes = np.sort(
        np.r_[
            np.linspace(-3, 6, 200000),
            bins,
            np.nextafter(bins, -np.inf),
            np.nextafter(bins, np.inf),
        ]
    )
    events = np.array([0, 0, 1, 2])
    expected = _reference_hist(spikes, events)
    sizes = []
    histogram = np.histogram

    def bounded_histogram(values, **kwargs):
        sizes.append(len(values))
        return histogram(values, **kwargs)

    monkeypatch.setattr(event.np, 'histogram', bounded_histogram)
    x, actual = event._peri_event_histogram(spikes, events)
    assert_array_equal(actual, expected)
    assert_array_equal(x, bins[:-1])
    assert max(sizes) <= 65536
    assert np.isnan(event._peri_event_histogram(spikes, [])[1]).all()
    assert_array_equal(event._peri_event_histogram(np.array([]), [1])[1], 0)
    with pytest.raises(ValueError, match='finite'):
        event._peri_event_histogram(spikes, [np.nan])


def test_event_live_merge_split_undo_axes(event, monkeypatch, caplog):
    monkeypatch.setattr(event.EventView, 'plot_canvas_class', _Canvas)
    controller = _controller()
    timestamps = np.arange(16, dtype=np.uint32)
    events = {name: np.array([0.5]) for name in event.EVENT_NAMES}
    view = event.EventView(controller, events, timestamps, 10)
    view.on_select([0])
    initial_callbacks = len(view.canvas.figure.canvas.callbacks.callbacks['scroll_event'])
    axis, line = view.canvas.ax, view.canvas.ax.lines[0]
    view.on_select([1])
    assert view.canvas.ax is axis and view.canvas.ax.lines[0] is line
    clustering = controller.supervisor.clustering
    merged = clustering.merge([0, 1]).added[0]
    assert_allclose(event._cluster_times(controller, merged, timestamps, 10), [0.2, 0.5, 0.8, 1])
    view.on_select([merged])
    split = clustering.split([0]).added
    view.on_select(split)
    assert len(view.canvas.axes) == len(split)
    clustering.undo()
    assert_allclose(event._cluster_times(controller, merged, timestamps, 10), [0.2, 0.5, 0.8, 1])
    clustering.undo()
    assert_allclose(event._cluster_times(controller, 0, timestamps, 10), [0.2, 0.5])
    view.on_select([])
    assert len(view.canvas.figure.canvas.callbacks.callbacks['scroll_event']) == initial_callbacks
    assert not len(view.canvas.ax.lines[0].get_xdata())
    controller.model.spike_samples[0] = -1
    assert_allclose(event._cluster_times(controller, 0, timestamps, 10), [0.5])
    assert 'outside the recording clock' in caplog.text


def test_analyzer_path_preferences(waveform, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('PHY_WAVEFORM_ANALYZER', raising=False)
    controller = _controller()
    assert waveform._analyzer_path(controller) == tmp_path.parent / 'sorting_analyzer'
    preferred = tmp_path.parent / 'sorting_analyzer_after_phy'
    preferred.mkdir(exist_ok=True)
    assert waveform._analyzer_path(controller) == preferred
    monkeypatch.setenv('PHY_WAVEFORM_ANALYZER', 'custom')
    assert waveform._analyzer_path(controller) == tmp_path / 'custom'
    controller.waveform_analyzer_path = tmp_path / 'explicit'
    assert waveform._analyzer_path(controller) == controller.waveform_analyzer_path


class _Sorting:
    sampling_frequency = 1000

    def __init__(self):
        self.trains = {20: np.array([2, 5]), 3: np.array([8, 10])}
        self.calls = []

    def get_num_segments(self):
        return 1

    def count_num_spikes_per_unit(self):
        return {unit: len(train) for unit, train in self.trains.items()}

    def get_unit_spike_train(self, unit_id):
        self.calls.append(unit_id)
        return self.trains[unit_id]


def _fake_analyzer(sorting=None):
    extension = SimpleNamespace(get_data=lambda **kw: np.ones((2, 3, 2)))
    return SimpleNamespace(sorting=sorting or _Sorting(), get_extension=lambda name: extension)


def test_exact_matching_and_cached_candidates(waveform, caplog):
    sorting = _Sorting()
    source = waveform._SavedWaveforms(_fake_analyzer(sorting), 1000)
    assert source.match(np.array([5, 2]), 3) == 20  # Numerical IDs explicitly disagree.
    sorting.calls.clear()
    assert source.match(np.array([2, 5]), 3) == 20
    assert sorting.calls == [20]  # Does not rescan all equal-count trains.
    assert source.match(np.array([2, 6]), 20) is None
    assert source.match(np.array([2, 5, 8, 10]), 21) is None  # live merge
    assert source.match(np.array([2]), 22) is None  # live split
    assert source.match(np.array([2, 5]), 20) == 20  # undo
    assert source.match(np.array([], dtype=int), 0) is None
    assert 'omitting saved waveform' in caplog.text
    sorting.trains[21] = np.array([2, 5])
    ambiguous = waveform._SavedWaveforms(_fake_analyzer(sorting), 1000)
    assert ambiguous.match(np.array([2, 5]), 20) is None
    with pytest.raises(ValueError, match='sample rates'):
        waveform._SavedWaveforms(_fake_analyzer(), 30000)
    sorting.get_num_segments = lambda: 2
    with pytest.raises(ValueError, match='single-segment'):
        waveform._SavedWaveforms(_fake_analyzer(sorting), 1000)


def test_waveform_lazy_reuse(waveform, monkeypatch, tmp_path):
    spikeinterface = pytest.importorskip('spikeinterface')

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(waveform.WaveformSpikeinterfaceView, 'plot_canvas_class', _Canvas)
    controller = _controller()
    controller.waveform_analyzer_path = tmp_path
    calls = []

    def load(path, load_extensions):
        calls.append((path, load_extensions))
        return _fake_analyzer()

    monkeypatch.setattr(spikeinterface, 'load_sorting_analyzer', load)
    waveform.WaveformSpikeinterfaceViewPlugin().attach_to_controller(controller)
    assert not calls
    first = controller.view_creator['WaveformSpikeinterfaceView']()
    second = controller.view_creator['WaveformSpikeinterfaceView']()
    assert first.source is second.source
    assert calls == [(tmp_path, False)]


@pytest.mark.parametrize('save_templates', [False, True])
@pytest.mark.parametrize('sparse', [False, True])
def test_saved_analyzer_no_recording(waveform, tmp_path, monkeypatch, save_templates, sparse):
    si = pytest.importorskip('spikeinterface')
    from probeinterface import Probe

    rng = np.random.default_rng(3)
    recording = si.NumpyRecording(rng.normal(size=(1000, 4)).astype('float32'), 1000)
    probe = Probe(ndim=2)
    probe.set_contacts(
        np.array([[0, 0], [0, 20], [20, 0], [20, 20]]), shapes='circle', shape_params={'radius': 5}
    )
    probe.set_device_channel_indices(np.arange(4))
    recording = recording.set_probe(probe)
    sorting = si.NumpySorting.from_unit_dict(
        {20: np.array([200, 400]), 3: np.array([600, 800])}, sampling_frequency=1000
    )
    sparsity = (
        si.ChannelSparsity(
            np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=bool),
            sorting.unit_ids,
            recording.channel_ids,
        )
        if sparse
        else None
    )
    folder = tmp_path / 'analyzer'
    analyzer = si.create_sorting_analyzer(
        sorting, recording, folder=folder, format='binary_folder', sparse=sparse, sparsity=sparsity
    )
    analyzer.compute('random_spikes', max_spikes_per_unit=10)
    analyzer.compute('waveforms', ms_before=2, ms_after=3)
    expected = (
        analyzer.get_extension('waveforms').get_waveforms_one_unit(20, force_dense=True).mean(0)
    )
    if save_templates:
        analyzer.compute('templates', operators=['average'])
    # Synthetic recording is unserializable; ensure no recording metadata exists on reload.
    for name in ('recording.json', 'recording.pickle'):
        (folder / name).unlink(missing_ok=True)
    loaded = si.load_sorting_analyzer(folder, load_extensions=False)
    assert not loaded.has_recording()
    assert loaded.extensions == {}

    def forbid_recording(self):
        pytest.fail('Requested raw recording')

    monkeypatch.setattr(type(loaded), 'recording', property(forbid_recording))
    source = waveform._SavedWaveforms(loaded, 1000)
    assert source.from_waveforms is not save_templates
    templates = source.templates([20])
    assert_allclose(templates.to_dense().templates_array[0], expected)
    assert 'noise_levels' not in loaded.extensions
    controller = _controller(samples=(200, 400, 600, 800))
    monkeypatch.setattr(waveform.WaveformSpikeinterfaceView, 'plot_canvas_class', _Canvas)
    view = waveform.WaveformSpikeinterfaceView(source, controller)
    view.on_select([0, 1])
    assert len(view.canvas.ax.lines) == 2
    assert view.canvas.ax.get_legend().texts[0].get_text() == 'Phy 0 (analyzer 20)'
    merged = controller.supervisor.clustering.merge([0, 1]).added[0]
    view.on_select([merged])
    assert not view.canvas.ax.lines
    controller.supervisor.clustering.undo()
    view.on_select([0])
    assert len(view.canvas.ax.lines) == 1
    view.on_select([])
    assert not view.canvas.ax.lines
