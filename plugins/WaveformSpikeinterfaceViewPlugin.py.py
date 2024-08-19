"""Show how to create a custom matplotlib view in the GUI."""

from phy import IPlugin
from phy.cluster.views import ManualClusteringView  # Base class for phy views
from phy.plot.plot import PlotCanvasMpl  # matplotlib canvas
import os
from pathlib import Path
from spikeinterface import load_sorting_analyzer
import spikeinterface.widgets as sw

class WaveformSpikeinterfaceView(ManualClusteringView):
    plot_canvas_class = PlotCanvasMpl  # use matplotlib instead of OpenGL (the default)

    def __init__(self, analyzer=None):
        """features is a function (cluster_id => Bunch(data, ...)) where data is a 3D array."""
        super(WaveformSpikeinterfaceView, self).__init__()
        self.analyzer = analyzer

    def on_select(self, cluster_ids=(), **kwargs):
        self.cluster_ids = cluster_ids
        # We don't display anything if no clusters are selected.
        if not cluster_ids:
            return

        # To simplify, we only consider the first PC component of the first 2 best channels.
        # Note that the features are in sparse format, where data's shape is
        # (n_spikes, n_best_channels, n_pcs). Only best channels for that clusters are
        # considered.
        # For this example, we just take the first 2 dimensions.
        #x, y = self.features(cluster_ids[0]).data[:, :2, 0].T

        self.canvas.ax.clear()


        sw.plot_unit_waveforms(self.analyzer, unit_ids=cluster_ids, ax=self.canvas.ax, same_axis=True, plot_waveforms=False,
                                   alpha_templates=0.5, shade_templates=False, plot_legend=False)



        # Use this to update the matplotlib figure.
        self.canvas.update()

        self.legend = self.canvas.ax.legend()



# Get the current working directory
current_dir = Path(os.getcwd())

analyzer_path = Path(*current_dir.parts[:-1]) / "analyzer"

class WaveformSpikeinterfaceViewPlugin(IPlugin):
    def attach_to_controller(self, controller):
        def create_waveform_view():
            """A function that creates and returns a view."""
            return WaveformSpikeinterfaceView(analyzer=load_sorting_analyzer(analyzer_path))

        controller.view_creator['WaveformSpikeinterfaceView'] = create_waveform_view