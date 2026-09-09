"""Display saved analyzer mean waveforms, matched to live Phy spike trains."""

import hashlib
import logging
import os
from pathlib import Path

import numpy as np

from phy import IPlugin
from phy.cluster.views import ManualClusteringView
from phy.plot.plot import PlotCanvasMpl

logger = logging.getLogger(__name__)


def _analyzer_path(controller):
    current_dir = Path.cwd()
    explicit = getattr(controller, 'waveform_analyzer_path', None) or os.environ.get(
        'PHY_WAVEFORM_ANALYZER'
    )
    if explicit is not None:
        path = Path(explicit).expanduser()
        return path if path.is_absolute() else current_dir / path
    preferred = current_dir.parent / 'sorting_analyzer_after_phy'
    return preferred if preferred.exists() else current_dir.parent / 'sorting_analyzer'


class _SavedWaveforms:
    def __init__(self, analyzer, sample_rate):
        self.analyzer = analyzer
        sorting = analyzer.sorting
        if sorting.get_num_segments() != 1:
            raise ValueError('Waveform matching requires a single-segment analyzer')
        if sorting.sampling_frequency != sample_rate:
            raise ValueError(
                'Analyzer and Phy sample rates differ; cannot match sample coordinates'
            )
        self.extension = analyzer.get_extension('templates')
        self._templates = None
        if self.extension is not None:
            try:
                # get_templates() can compute/save missing operators; get_data() cannot.
                self._templates = self.extension.get_data(operator='average')
            except ValueError:
                logger.warning('No saved average templates; trying saved waveforms instead.')
        self.from_waveforms = self._templates is None
        if self.from_waveforms:
            self.extension = analyzer.get_extension('waveforms')
        if self.extension is None:
            raise ValueError(
                'Analyzer needs saved average templates or waveforms; raw data is not read'
            )
        # Counts are immutable analyzer-side metadata, not a cache of live Phy assignments.
        self._units_by_count = {}
        self._match_indices = {}
        for unit_id, count in sorting.count_num_spikes_per_unit().items():
            self._units_by_count.setdefault(count, []).append(unit_id)

    @staticmethod
    def _signature(samples):
        return hashlib.blake2b(np.asarray(samples, dtype='<i8').tobytes()).digest()

    def match(self, samples, cluster_id):
        samples = np.sort(np.asarray(samples))
        if samples.dtype.kind not in 'iu' or not len(samples):
            logger.warning('Cluster %s: no nonempty integer spike train to match.', cluster_id)
            return None
        count = len(samples)
        if count not in self._match_indices:
            index = {}
            for unit_id in self._units_by_count.get(count, ()):
                train = self.analyzer.sorting.get_unit_spike_train(unit_id)
                index.setdefault(self._signature(train), []).append(unit_id)
            self._match_indices[count] = index
        matches = [
            unit_id
            for unit_id in self._match_indices[count].get(self._signature(samples), ())
            if np.array_equal(samples, self.analyzer.sorting.get_unit_spike_train(unit_id))
        ]
        if len(matches) != 1:
            logger.warning(
                'Cluster %s: %d exact analyzer spike-train matches; omitting saved waveform. '
                'Check analyzer/sample origin or regenerate it after merges/splits.',
                cluster_id,
                len(matches),
            )
            return None
        return matches[0]

    def templates(self, unit_ids):
        """Build only the selected means; never compute an extension or read raw traces."""
        from spikeinterface.core import Templates

        analyzer = self.analyzer
        if not self.from_waveforms:
            means = self._templates[analyzer.sorting.ids_to_indices(unit_ids)]
        else:
            means = []
            for unit_id in unit_ids:
                waveforms = self.extension.get_waveforms_one_unit(unit_id, force_dense=False)
                if not len(waveforms):
                    raise ValueError(f'Analyzer unit {unit_id} has no saved waveforms')
                mean = waveforms.mean(axis=0)
                if analyzer.sparsity is not None:
                    dense = np.zeros((mean.shape[0], len(analyzer.channel_ids)), dtype=mean.dtype)
                    dense[:, analyzer.sparsity.unit_id_to_channel_indices[unit_id]] = mean
                    mean = dense
                means.append(mean)
            means = np.asarray(means)
        templates = Templates(
            templates_array=means,
            sampling_frequency=analyzer.sampling_frequency,
            nbefore=self.extension.nbefore,
            channel_ids=analyzer.channel_ids,
            unit_ids=np.asarray(unit_ids),
            probe=analyzer.get_probe(),
            is_in_uV=analyzer.return_in_uV,
        )
        if analyzer.sparsity is not None:
            indices = analyzer.sorting.ids_to_indices(unit_ids)
            templates = templates.to_sparse(analyzer.sparsity.mask[indices])
        return templates


class WaveformSpikeinterfaceView(ManualClusteringView):
    plot_canvas_class = PlotCanvasMpl

    def __init__(self, source, controller):
        super().__init__()
        self.source = source
        self.controller = controller

    def on_select(self, cluster_ids=(), **kwargs):
        import spikeinterface.widgets as sw

        self.cluster_ids = list(cluster_ids)
        ax = self.canvas.ax
        ax.clear()
        matches = []
        labels = []
        for cluster_id in cluster_ids:
            spike_ids = self.controller.supervisor.clustering.spikes_per_cluster.get(
                cluster_id, ()
            )
            samples = self.controller.model.spike_samples[np.asarray(spike_ids, dtype=np.int64)]
            unit_id = self.source.match(samples, cluster_id)
            if unit_id is not None:
                matches.append(unit_id)
                labels.append(f'Phy {cluster_id} (analyzer {unit_id})')
        if matches:
            templates = self.source.templates(matches)
            sw.plot_unit_waveforms(
                templates,
                ax=ax,
                same_axis=True,
                plot_waveforms=False,
                plot_templates=True,
                alpha_templates=0.5,
                shade_templates=False,
                templates_percentile_shading=None,
                plot_legend=False,
                set_title=False,
                backend='matplotlib',
            )
            ax.legend(labels=labels)
            ax.set_title('Saved analyzer mean waveforms')
        elif cluster_ids:
            ax.set_title('No unambiguous saved waveform for selected clusters')
        self.canvas.update()


class WaveformSpikeinterfaceViewPlugin(IPlugin):
    def attach_to_controller(self, controller):
        source = None
        source_path = None

        def create_waveform_view():
            nonlocal source, source_path
            path = '<unresolved analyzer path>'
            try:
                path = _analyzer_path(controller)
                if not path.is_dir():
                    raise FileNotFoundError(f'No saved sorting analyzer at {path}')
                from spikeinterface import load_sorting_analyzer

                controller.supervisor.clustering.spikes_per_cluster
                controller.model.spike_samples
                if source is None or source_path != path:
                    analyzer = load_sorting_analyzer(path, load_extensions=False)
                    source = _SavedWaveforms(analyzer, controller.model.sample_rate)
                    source_path = path
                return WaveformSpikeinterfaceView(source, controller)
            except Exception:
                # Optional GUI plugin boundary: do not abort startup for unavailable analyzers.
                logger.warning(
                    'Cannot create WaveformSpikeinterfaceView from %s.', path, exc_info=True
                )
                return None

        controller.view_creator['WaveformSpikeinterfaceView'] = create_waveform_view
