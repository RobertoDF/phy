# API documentation of phy

phy: interactive visualization and manual spike sorting of large-scale ephys data.

## Table of contents

### [phy.utils](#phyutils)

* [phy.utils.add_alpha](#phyutilsadd_alphac-alpha10)
* [phy.utils.attach_plugins](#phyutilsattach_pluginscontroller-pluginsnone-config_dirnone)
* [phy.utils.ensure_dir_exists](#phyutilsensure_dir_existspath)
* [phy.utils.load_json](#phyutilsload_jsonpath)
* [phy.utils.load_master_config](#phyutilsload_master_configconfig_dirnone)
* [phy.utils.load_pickle](#phyutilsload_picklepath)
* [phy.utils.phy_config_dir](#phyutilsphy_config_dir)
* [phy.utils.read_python](#phyutilsread_pythonpath)
* [phy.utils.read_text](#phyutilsread_textpath)
* [phy.utils.read_tsv](#phyutilsread_tsvpath)
* [phy.utils.save_json](#phyutilssave_jsonpath-data)
* [phy.utils.save_pickle](#phyutilssave_picklepath-data)
* [phy.utils.selected_cluster_color](#phyutilsselected_cluster_colori-alpha10)
* [phy.utils.write_text](#phyutilswrite_textpath-contents)
* [phy.utils.write_tsv](#phyutilswrite_tsvpath-data-first_fieldnone-exclude_fields-n_significant_figures4)
* [phy.utils.Bunch](#phyutilsbunch)
* [phy.utils.ClusterColorSelector](#phyutilsclustercolorselector)
* [phy.utils.Context](#phyutilscontext)
* [phy.utils.IPlugin](#phyutilsiplugin)


### [phy.gui](#phygui)

* [phy.gui.busy_cursor](#phyguibusy_cursor)
* [phy.gui.create_app](#phyguicreate_app)
* [phy.gui.input_dialog](#phyguiinput_dialogtitle-sentence-textnone)
* [phy.gui.is_high_dpi](#phyguiis_high_dpi)
* [phy.gui.message_box](#phyguimessage_boxmessage-titlemessage-levelnone)
* [phy.gui.prompt](#phyguipromptmessage-buttonsyes-no-titlequestion)
* [phy.gui.require_qt](#phyguirequire_qtfunc)
* [phy.gui.run_app](#phyguirun_app)
* [phy.gui.screen_size](#phyguiscreen_size)
* [phy.gui.screenshot](#phyguiscreenshotwidget-path)
* [phy.gui.thread_pool](#phyguithread_pool)
* [phy.gui.Actions](#phyguiactions)
* [phy.gui.Debouncer](#phyguidebouncer)
* [phy.gui.GUI](#phyguigui)
* [phy.gui.GUIState](#phyguiguistate)
* [phy.gui.HTMLBuilder](#phyguihtmlbuilder)
* [phy.gui.HTMLWidget](#phyguihtmlwidget)
* [phy.gui.IPythonView](#phyguiipythonview)
* [phy.gui.Snippets](#phyguisnippets)
* [phy.gui.Table](#phyguitable)
* [phy.gui.Worker](#phyguiworker)


### [phy.plot](#phyplot)

* [phy.plot.get_linear_x](#phyplotget_linear_xn_signals-n_samples)
* [phy.plot.Axes](#phyplotaxes)
* [phy.plot.AxisLocator](#phyplotaxislocator)
* [phy.plot.BaseCanvas](#phyplotbasecanvas)
* [phy.plot.BaseLayout](#phyplotbaselayout)
* [phy.plot.BaseVisual](#phyplotbasevisual)
* [phy.plot.BatchAccumulator](#phyplotbatchaccumulator)
* [phy.plot.Boxed](#phyplotboxed)
* [phy.plot.GLSLInserter](#phyplotglslinserter)
* [phy.plot.Grid](#phyplotgrid)
* [phy.plot.HistogramVisual](#phyplothistogramvisual)
* [phy.plot.ImageVisual](#phyplotimagevisual)
* [phy.plot.Lasso](#phyplotlasso)
* [phy.plot.LineVisual](#phyplotlinevisual)
* [phy.plot.PanZoom](#phyplotpanzoom)
* [phy.plot.PlotCanvas](#phyplotplotcanvas)
* [phy.plot.PlotVisual](#phyplotplotvisual)
* [phy.plot.PolygonVisual](#phyplotpolygonvisual)
* [phy.plot.Range](#phyplotrange)
* [phy.plot.Scale](#phyplotscale)
* [phy.plot.ScatterVisual](#phyplotscattervisual)
* [phy.plot.Subplot](#phyplotsubplot)
* [phy.plot.TextVisual](#phyplottextvisual)
* [phy.plot.TransformChain](#phyplottransformchain)
* [phy.plot.Translate](#phyplottranslate)
* [phy.plot.UniformPlotVisual](#phyplotuniformplotvisual)
* [phy.plot.UniformScatterVisual](#phyplotuniformscattervisual)


### [phy.cluster](#phycluster)

* [phy.cluster.select_traces](#phyclusterselect_tracestraces-interval-sample_ratenone)
* [phy.cluster.ClusterMeta](#phyclusterclustermeta)
* [phy.cluster.ClusterView](#phyclusterclusterview)
* [phy.cluster.Clustering](#phyclusterclustering)
* [phy.cluster.CorrelogramView](#phyclustercorrelogramview)
* [phy.cluster.FeatureView](#phyclusterfeatureview)
* [phy.cluster.HistogramView](#phyclusterhistogramview)
* [phy.cluster.ManualClusteringView](#phyclustermanualclusteringview)
* [phy.cluster.ProbeView](#phyclusterprobeview)
* [phy.cluster.RasterView](#phyclusterrasterview)
* [phy.cluster.ScatterView](#phyclusterscatterview)
* [phy.cluster.SimilarityView](#phyclustersimilarityview)
* [phy.cluster.Supervisor](#phyclustersupervisor)
* [phy.cluster.TemplateView](#phyclustertemplateview)
* [phy.cluster.TraceView](#phyclustertraceview)
* [phy.cluster.UpdateInfo](#phyclusterupdateinfo)
* [phy.cluster.WaveformView](#phyclusterwaveformview)


### [phy.apps.template](#phyappstemplate)

* [phy.apps.template.template_describe](#phyappstemplatetemplate_describeparams_path)
* [phy.apps.template.template_gui](#phyappstemplatetemplate_guiparams_path)
* [phy.apps.template.TemplateController](#phyappstemplatetemplatecontroller)
* [phy.apps.template.TemplateModel](#phyappstemplatetemplatemodel)




## phy.utils

Utilities: plugin system, event system, configuration system, profiling, debugging, cacheing,
basic read/write functions.

---


**`phy.utils.add_alpha(c, alpha=1.0)`**

Add an alpha channel to an RGB color.

**Parameters**


* `c : array-like (2D, shape[1] == 3) or 3-tuple` 　 

* `alpha : float` 　 

---


**`phy.utils.attach_plugins(controller, plugins=None, config_dir=None)`**

Attach plugins to a controller object.

Attached plugins are those found in the user configuration file for the given gui_name or
class name of the Controller instance, plus those specified in the plugins keyword argument.

**Parameters**


* `controller : object` 　 
    The controller object that will be passed to the `attach_to_controller()` plugins methods.

* `plugins : list` 　 
    List of plugins to attach in addition to those found in the user configuration file.

* `config_dir : str` 　 
    Path to the user configuration file. By default, the directory is `~/.phy/`.

---


**`phy.utils.ensure_dir_exists(path)`**

Ensure a directory exists, and create it otherwise.

---


**`phy.utils.load_json(path)`**

Load a JSON file.

---


**`phy.utils.load_master_config(config_dir=None)`**

Load a master Config file from the user configuration file (by default, this is
`~/.phy/phy_config.py`).

---


**`phy.utils.load_pickle(path)`**

Load a pickle file using joblib.

---


**`phy.utils.phy_config_dir()`**

Return the absolute path to the phy user directory. By default, `~/.phy/`.

---


**`phy.utils.read_python(path)`**

Read a Python file.

**Parameters**


* `path : str or Path` 　 

**Returns**


* `metadata : dict` 　 
    A dictionary containing all variables defined in the Python file (with `exec()`).

---


**`phy.utils.read_text(path)`**

Read a text file.

---


**`phy.utils.read_tsv(path)`**

Read a CSV/TSV file.

**Returns**


* `data : list of dicts` 　 

---


**`phy.utils.save_json(path, data)`**

Save a dictionary to a JSON file.

Support NumPy arrays and QByteArray objects. NumPy arrays are saved as base64-encoded strings,
except for 1D arrays with less than 10 elements, which are saved as a list for human
readability.

---


**`phy.utils.save_pickle(path, data)`**

Save data to a pickle file using joblib.

---


**`phy.utils.selected_cluster_color(i, alpha=1.0)`**

Return the color, as a 4-tuple, of the i-th selected cluster.

---


**`phy.utils.write_text(path, contents)`**

Write a text file.

---


**`phy.utils.write_tsv(path, data, first_field=None, exclude_fields=(), n_significant_figures=4)`**

Write a CSV/TSV file.

**Parameters**


* `data : list of dicts` 　 

* `first_field : str` 　 
    The name of the field that should come first in the file.

* `exclude_fields : list-like` 　 
    Fields present in the data that should not be saved in the file.

* `n_significant_figures : int` 　 
    Number of significant figures used for floating-point numbers in the file.

---

### phy.utils.Bunch

A subclass of dictionary with an additional dot syntax.

---


**`Bunch.copy(self)`**

Return a new Bunch instance which is a copy of the current Bunch instance.

---

### phy.utils.ClusterColorSelector

Assign a color to clusters depending on cluster labels or metrics.

---


**`ClusterColorSelector.get(self, cluster_id, alpha=None)`**

Return the RGBA color of a single cluster.

---


**`ClusterColorSelector.get_colors(self, cluster_ids, alpha=1.0)`**

Return the RGBA colors of some clusters.

---


**`ClusterColorSelector.get_values(self, cluster_ids)`**

Get the values of clusters for the selected color field..

---


**`ClusterColorSelector.map(self, values)`**

Convert values to colors using the selected colormap.

**Parameters**


* `values : array-like (1D)` 　 

**Returns**


* `colors : array-like (2D, shape[1] == 3)` 　 

---


**`ClusterColorSelector.set_cluster_ids(self, cluster_ids)`**

Precompute the value range for all clusters.

---


**`ClusterColorSelector.set_color_mapping(self, color_field=None, colormap=None, categorical=None, logarithmic=None)`**

Set the field used to choose the cluster colors, and the associated colormap.

**Parameters**


* `color_field : str` 　 
    Name of the cluster metrics or label to use for the color.

* `colormap : array-like` 　 
    A `(N, 3)` array with the colormaps colors

* `categorical : boolean` 　 
    Whether the colormap is categorical (one value = one color) or continuous (values
    are continuously mapped from their initial interval to the colors).

* `logarithmic : boolean` 　 
    Whether to use a logarithmic transform for the mapping.

---


**`ClusterColorSelector.set_state(self, state)`**

Set the colormap state.

---


**`ClusterColorSelector.state`**

Colormap state. This is a Bunch with the following keys: color_field, colormap,
categorical, logarithmic.

---

### phy.utils.Context

Handle function disk and memory caching with joblib.

Memcaching a function is used to save *in memory* the output of the function for all
passed inputs. Input should be hashable. NumPy arrays are supported. The contents of the
memcache in memory can be persisted to disk with `context.save_memcache()` and
`context.load_memcache()`.

Caching a function is used to save *on disk* the output of the function for all passed
inputs. Input should be hashable. NumPy arrays are supported. This is to be preferred
over memcache when the inputs or outputs are large, and when the computations are longer
than loading the result from disk.

**Constructor**


* `cache_dir : str` 　 
    The directory in which the cache will be created.

* `verbose : int` 　 
    The verbosity level passed to joblib Memory.

**Examples**

```python
@context.memcache
def my_function(x):
    return x * x

@context.cache
def my_function(x):
    return x * x
```

---


**`Context.cache(self, f)`**

Cache a function using the context's cache directory.

---


**`Context.load(self, name, location='local')`**

Load a dictionary saved in the cache directory.

**Parameters**


* `name : str` 　 
    The name of the object to save to disk.

* `location : str` 　 
    Can be `local` or `global`.

---


**`Context.load_memcache(self, name)`**

Load the memcache from disk (pickle file), if it exists.

---


**`Context.memcache(self, f)`**

Cache a function in memory using an internal dictionary.

---


**`Context.save(self, name, data, location='local', kind='json')`**

Save a dictionary in a JSON/pickle file within the cache directory.

**Parameters**


* `name : str` 　 
    The name of the object to save to disk.

* `data : dict` 　 
    Any serializable dictionary that will be persisted to disk.

* `location : str` 　 
    Can be `local` or `global`.

* `kind : str` 　 
    Can be `json` or `pickle`.

---


**`Context.save_memcache(self)`**

Save the memcache to disk using pickle.

---

### phy.utils.IPlugin

All plugin classes should derive from this class.

Plugin classes should just implement a method `attach_to_controller(self, controller)`.

---

## phy.gui

GUI routines.

---


**`phy.gui.busy_cursor()`**

Context manager displaying a busy cursor during a long command.

---


**`phy.gui.create_app()`**

Create a Qt application.

---


**`phy.gui.input_dialog(title, sentence, text=None)`**

Display a dialog with a text box.

**Parameters**


* `title : str` 　 
    Title of the dialog.

* `sentence : str` 　 
    Message of the dialog.

* `text : str` 　 
    Default text in the text box.

---


**`phy.gui.is_high_dpi()`**

Return whether the screen has a high density.

Note: currently, this only returns whether the screen width is greater than an arbitrary
value chosen at 3000.

---


**`phy.gui.message_box(message, title='Message', level=None)`**

Display a message box.

**Parameters**

* `message : str` 　 

* `title : str` 　 

* `level : str` 　 
    information, warning, or critical

---


**`phy.gui.prompt(message, buttons=('yes', 'no'), title='Question')`**

Display a dialog with several buttons to confirm or cancel an action.

**Parameters**


* `message : str` 　 
    Dialog message.

* `buttons : tuple` 　 
    Name of the standard buttons to show in the prompt: yes, no, ok, cancel, close, etc.
    See the full list at https://doc.qt.io/qt-5/qmessagebox.html#StandardButton-enum

* `title : str` 　 
    Dialog title.

---


**`phy.gui.require_qt(func)`**

Function decorator to specify that a function requires a Qt application.

Use this decorator to specify that a function needs a running
Qt application before it can run. An error is raised if that is not
the case.

---


**`phy.gui.run_app()`**

Run the Qt application.

---


**`phy.gui.screen_size()`**

Return the screen size as a tuple (width, height).

---


**`phy.gui.screenshot(widget, path)`**

Save a screenshot of a Qt widget to a PNG file.

**Parameters**


* `widget : Qt widget` 　 
    Any widget to capture (including OpenGL widgets).

* `path : str or Path` 　 
    Path to the PNG file.

---


**`phy.gui.thread_pool()`**

Return a QThreadPool instance that can `start()` Worker instances for multithreading.

**Example**

```python
w = Worker(print, "hello world")
thread_pool().start(w)
```

---

### phy.gui.Actions

Group of actions bound to a GUI.

This class attaches to a GUI and implements the following features:

* Add and remove actions
* Keyboard shortcuts for the actions
* Display all shortcuts

**Constructor**


* `gui : GUI instance` 　 

* `name : str` 　 
    Name of this group of actions.

* `menu : str` 　 
    Name of the GUI menu that will contain the actions.

* `submenu : str` 　 
    Name of the GUI submenu that will contain the actions.

* `default_shortcuts : dict` 　 
    Map action names to keyboard shortcuts (regular strings).

---


**`Actions.add(self, callback=None, name=None, shortcut=None, alias=None, prompt=False, n_args=None, docstring=None, menu=None, submenu=None, verbose=True, checkable=False, checked=False, prompt_default=None, show_shortcut=True)`**

Add an action with a keyboard shortcut.

**Parameters**


* `callback : function` 　 
    Take no argument if checkable is False, or a boolean (checked) if it is True

* `name : str` 　 
    Action name, the callback's name by default.

* `shortcut : str` 　 
    The keyboard shortcut for this action.

* `alias : str` 　 
    Snippet, the name by default.

* `prompt : boolean` 　 
    Whether this action should display a dialog with an input box where the user can
    write arguments to the callback function.

* `n_args : int` 　 
    If prompt is True, specify the number of expected arguments.

* `prompt_default : str` 　 
    The default text in the input text box, if prompt is True.

* `docstring : str` 　 
    The action docstring, to be displayed in the status bar when hovering over the action
    item in the menu. By default, the function's docstring.

* `menu : str` 　 
    The name of the menu where the action should be added. It is automatically created
    if it doesn't exist.

* `submenu : str` 　 
    The name of the submenu where the action should be added. It is automatically created
    if it doesn't exist.

* `checkable : boolean` 　 
    Whether the action is checkable (toggle on/off).

* `checked : boolean` 　 
    Whether the checkable action is initially checked or not.

* `show_shortcut : boolean` 　 
    Whether to show the shortcut in the Help action that displays all GUI shortcuts.

---


**`Actions.disable(self, name=None)`**

Disable all actions, or only one if a name is passed.

---


**`Actions.enable(self, name=None)`**

Enable all actions, or only one if a name is passed..

---


**`Actions.get(self, name)`**

Get a QAction instance from its name.

---


**`Actions.remove(self, name)`**

Remove an action.

---


**`Actions.remove_all(self)`**

Remove all actions.

---


**`Actions.run(self, name, *args)`**

Run an action as specified by its name.

---


**`Actions.separator(self, menu=None)`**

Add a separator.

**Parameters**


* `menu : str` 　 
    The menu that will contain the separator, or the Actions menu by default.

---


**`Actions.show_shortcuts(self)`**

Display all shortcuts in the console.

---


**`Actions.shortcuts`**

A dictionary mapping action names to keyboard shortcuts.

---

### phy.gui.Debouncer

Debouncer to work in a Qt application.

Jobs are submitted at given times. They are executed immediately if the
delay since the last submission is greater than some threshold. Otherwise, execution
is delayed until the delay since the last submission is greater than the threshold.
During the waiting time, all submitted jobs erase previous jobs in the queue, so
only the last jobs are taken into account.

This is used when multiple row selections are done in an HTML table, and each row
selection is taking a perceptible time to finish.

**Constructor**


* `delay : int` 　 
    The minimal delay between the execution of two successive actions.

**Example**

```python
d = Debouncer(delay=250)
for i in range(10):
    d.submit(print, "hello world", i)
d.trigger()  # show "hello world 0" and "hello world 9" after a delay

```

---


**`Debouncer.submit(self, f, *args, key=None, **kwargs)`**

Submit a function call. Execute immediately if the delay since the last submission
is higher than the threshold, or wait until executing it otherwiser.

---


**`Debouncer.trigger(self)`**

Execute the pending actions.

---

### phy.gui.GUI

A Qt main window containing docking widgets. This class derives from `QMainWindow`.

**Constructor**


* `position : 2-tuple` 　 
    Coordinates of the GUI window on the screen, in pixels.

* `size : 2-tuple` 　 
    Requested size of the GUI window, in pixels.

* `name : str` 　 
    Name of the GUI window, set in the title bar.

* `subtitle : str` 　 
    Subtitle of the GUI window, set in the title bar after the name.

* `view_creator : dict` 　 
    Map view classnames to functions that take no arguments and return a new view instance
    of that class.

* `view_count : dict` 　 
    Map view classnames to integers specifying the number of views to create for every
    view class.

* `default_views : list-like` 　 
    List of view names to create by default (overriden by `view_count` if not empty).

* `config_dir : str or Path` 　 
    User configuration directory used to load/save the GUI state

**Events**

close
show
add_view
close_view

---


**`GUI.add_view(self, view, position=None, closable=True, floatable=True, floating=None)`**

Add a dock widget to the main window.

**Parameters**


* `view : View` 　 

* `position : str` 　 
    Relative position where to add the view (left, right, top, bottom).

* `closable : boolean` 　 
    Whether the view can be closed by the user.

* `floatable : boolean` 　 
    Whether the view can be detached from the main GUI.

* `floating : boolean` 　 
    Whether the view should be added in floating mode or not.

---


**`GUI.closeEvent(self, e)`**

Qt slot when the window is closed.

---


**`GUI.create_views(self)`**

Create and add as many views as specified in view_count.

---


**`GUI.dialog(self, message)`**

Show a message in a dialog box.

---


**`GUI.get_menu(self, name)`**

Get or create a menu.

---


**`GUI.get_submenu(self, menu, name)`**

Get or create a submenu.

---


**`GUI.get_view(self, cls, index=0)`**

Return a view from a given class. If there are multiple views of the same class,
specify the view index (0 by default).

---


**`GUI.list_views(self, cls)`**

Return the list of views deriving from a given class.

---


**`GUI.lock_status(self)`**

Lock the status bar.

---


**`GUI.remove_menu(self, name)`**

Remove a menu.

---


**`GUI.restore_geometry_state(self, gs)`**

Restore the position of the main window and the docks.

The GUI widgets need to be recreated first.

This function can be called in `on_show()`.

---


**`GUI.save_geometry_state(self)`**

Return picklable geometry and state of the window and docks.

This function can be called in `on_close()`.

---


**`GUI.set_default_actions(self)`**

Create the default actions (file, views, help...).

---


**`GUI.show(self)`**

Show the window.

---


**`GUI.unlock_status(self)`**

Unlock the status bar.

---


**`GUI.status_message`**

The message in the status bar, can be set by the user.

---


**`GUI.view_count`**

Return the number of views of every type, as a dictionary mapping view class names
to an integer.

---


**`GUI.views`**

Return the list of views in the GUI.

---

### phy.gui.GUIState

Represent the state of the GUI: positions of the views and all parameters associated
to the GUI and views. Derive from `Bunch`, which itself derives from `dict`.

The GUI state is automatically loaded from the user configuration directory.
The default path is `~/.phy/GUIName/state.json`.

---


**`GUIState.copy(self)`**

Return a new Bunch instance which is a copy of the current Bunch instance.

---


**`GUIState.get_view_state(self, view)`**

Return the state of a view instance.

---


**`GUIState.load(self)`**

Load the state from the JSON file in the config dir.

---


**`GUIState.save(self)`**

Save the state to the JSON files in the config dir (global) and local dir (if any).

---


**`GUIState.update_view_state(self, view, state)`**

Update the state of a view instance.

**Parameters**


* `view : View instance` 　 

* `state : Bunch instance` 　 

---

### phy.gui.HTMLBuilder

Build an HTML widget.

---


**`HTMLBuilder.add_header(self, s)`**

Add HTML headers.

---


**`HTMLBuilder.add_script(self, s)`**

Add Javascript code.

---


**`HTMLBuilder.add_script_src(self, filename)`**

Add a link to a Javascript file.

---


**`HTMLBuilder.add_style(self, s)`**

Add a CSS style.

---


**`HTMLBuilder.add_style_src(self, filename)`**

Add a link to a stylesheet URL.

---


**`HTMLBuilder.set_body(self, body)`**

Set the HTML body of the widget.

---


**`HTMLBuilder.set_body_src(self, filename)`**

Set the path to an HTML file containing the body of the widget.

---


**`HTMLBuilder.html`**

Return the reconstructed HTML code of the widget.

---

### phy.gui.HTMLWidget

An HTML widget that is displayed with Qt, with Javascript support and Python-Javascript
interactions capabilities. These interactions are asynchronous in Qt5, which requires
extensive use of callback functions in Python, as well as synchronization primitives
for unit tests.

**Constructor**


* `parent : Widget` 　 

* `title : window title` 　 

* `debounce_events : list-like` 　 
    The list of event names, raised by the underlying HTML widget, that should be debounced.

---


**`HTMLWidget.build(self, callback=None)`**

Rebuild the HTML code of the widget.

---


**`HTMLWidget.eval_js(self, expr, callback=None)`**

Evaluate a Javascript expression.

**Parameters**


* `expr : str` 　 
    A Javascript expression.

* `callback : function` 　 
    A Python function that is called once the Javascript expression has been
    evaluated. It takes as input the output of the Javascript expression.

---


**`HTMLWidget.set_html(self, html, callback=None)`**

Set the HTML code.

---


**`HTMLWidget.view_source(self, callback=None)`**

View the HTML source of the widget.

---

### phy.gui.IPythonView

A view with an IPython console living in the same Python process as the GUI.

---


**`IPythonView.attach(self, gui, **kwargs)`**

Add the view to the GUI, start the kernel, and inject the specified variables.

---


**`IPythonView.inject(self, **kwargs)`**

Inject variables into the IPython namespace.

---


**`IPythonView.start_kernel(self)`**

Start the IPython kernel.

---


**`IPythonView.stop(self)`**

Stop the kernel.

---

### phy.gui.Snippets

Provide keyboard snippets to quickly execute actions from a GUI.

This class attaches to a GUI and an `Actions` instance. To every command
is associated a snippet with the same name, or with an alias as indicated
in the action. The arguments of the action's callback functions can be
provided in the snippet's command with a simple syntax. For example, the
following command:

```
:my_action string 3-6
```

corresponds to:

```python
my_action('string', (3, 4, 5, 6))
```

The snippet mode is activated with the `:` keyboard shortcut. A snippet
command is activated with `Enter`, and one can leave the snippet mode
with `Escape`.

When the snippet mode is enabled (with `:`), this object adds a hidden Qt action
for every keystroke. These actions are removed when the snippet mode is disabled.

**Constructor**


* `gui : GUI instance` 　 

---


**`Snippets.is_mode_on(self)`**

Whether the snippet mode is enabled.

---


**`Snippets.mode_off(self)`**

Disable the snippet mode.

---


**`Snippets.mode_on(self)`**

Enable the snippet mode.

---


**`Snippets.run(self, snippet)`**

Execute a snippet command.

May be overridden.

---


**`Snippets.command`**

This is used to write a snippet message in the status bar. A cursor is appended at
the end.

---

### phy.gui.Table

A sortable table with support for selection. Derives from HTMLWidget.

This table uses the following Javascript implementation: https://github.com/kwikteam/tablejs
This Javascript class builds upon ListJS: https://listjs.com/

---


**`Table.add(self, objects)`**

Add objects object to the table.

---


**`Table.build(self, callback=None)`**

Rebuild the HTML code of the widget.

---


**`Table.change(self, objects)`**

Change some objects.

---


**`Table.eval_js(self, expr, callback=None)`**

Evaluate a Javascript expression.

The `table` Javascript variable can be used to interact with the underlying Javascript
table.

The table has sortable columns, a filter text box, support for single and multi selection
of rows. Rows can be skippable (used for ignored clusters in phy).

The table can raise Javascript events that are relayed to Python. Objects are
transparently serialized and deserialized in JSON. Basic types (numbers, strings, lists)
are transparently converted between Python and Javascript.

**Parameters**


* `expr : str` 　 
    A Javascript expression.

* `callback : function` 　 
    A Python function that is called once the Javascript expression has been
    evaluated. It takes as input the output of the Javascript expression.

---


**`Table.filter(self, text='')`**

Filter the view with a Javascript expression.

---


**`Table.first(self, callback=None)`**

Select the first item.

---


**`Table.get(self, id, callback=None)`**

Get the object given its id.

---


**`Table.get_current_sort(self, callback=None)`**

Get the current sort as a tuple `(name, dir)`.

---


**`Table.get_ids(self, callback=None)`**

Get the list of ids.

---


**`Table.get_next_id(self, callback=None)`**

Get the next non-skipped row id.

---


**`Table.get_previous_id(self, callback=None)`**

Get the previous non-skipped row id.

---


**`Table.get_selected(self, callback=None)`**

Get the currently selected rows.

---


**`Table.next(self, callback=None)`**

Select the next non-skipped row.

---


**`Table.previous(self, callback=None)`**

Select the previous non-skipped row.

---


**`Table.remove(self, ids)`**

Remove some objects from their ids.

---


**`Table.remove_all(self)`**

Remove all rows in the table.

---


**`Table.remove_all_and_add(self, objects)`**

Remove all rows in the table and add new objects.

---


**`Table.select(self, ids, callback=None, **kwargs)`**

Select some rows in the table from Python.

This function calls `table.select()` in Javascript, which raises a Javascript event
relayed to Python. This sequence of actions is the same when the user selects
rows directly in the HTML view.

---


**`Table.set_busy(self, busy)`**

Set the busy state of the GUI.

---


**`Table.set_html(self, html, callback=None)`**

Set the HTML code.

---


**`Table.sort_by(self, name, sort_dir='asc')`**

Sort by a given variable.

---


**`Table.view_source(self, callback=None)`**

View the HTML source of the widget.

---

### phy.gui.Worker

A task (just a Python function) running in the thread pool.

**Constructor**


* `fn : function` 　 

* `*args : function positional arguments` 　 

* `**kwargs : function keyword arguments` 　 

---


**`Worker.run(self)`**

Run the task. Should not be called directly unless you want to bypass the
thread pool.

---

## phy.plot

Plotting module based on OpenGL.

For advanced users!

---


**`phy.plot.get_linear_x(n_signals, n_samples)`**

Get a vertical stack of arrays ranging from -1 to 1.

Return a `(n_signals, n_samples)` array.

---

### phy.plot.Axes

Dynamic axes that move along the camera when panning and zooming.

**Constructor**


* `data_bounds : 4-tuple` 　 
    The data coordinates of the initial viewport (when there is no panning and zooming).

* `color : 4-tuple` 　 
    Color of the grid.

* `show_x : boolean` 　 
    Whether to show the vertical grid lines.

* `show_y : boolean` 　 
    Whether to show the horizontal grid lines.

---


**`Axes.attach(self, canvas)`**

Add the axes to a canvas.

Add the grid and text visuals to the canvas, and attach to the pan and zoom events
raised by the canvas.

---


**`Axes.reset_data_bounds(self, data_bounds, do_update=True)`**

Reset the bounds of the view in data coordinates.

Used when the view is recreated from scratch.

---


**`Axes.update_visuals(self)`**

Update the grid and text visuals after updating the axis locator.

---

### phy.plot.AxisLocator

Determine the location of ticks in a view.

**Constructor**


* `nbinsx : int` 　 
    Number of ticks on the x axis.

* `nbinsy : int` 　 
    Number of ticks on the y axis.

* `data_bounds : 4-tuple` 　 
    Initial coordinates of the viewport, as (xmin, ymin, xmax, ymax), in data coordinates.
    These are the data coordinates of the lower left and upper right points of the window.

---


**`AxisLocator.set_nbins(self, nbinsx=None, nbinsy=None)`**

Change the number of bins on the x and y axes.

---


**`AxisLocator.set_view_bounds(self, view_bounds=None)`**

Set the view bounds in normalized device coordinates. Used when panning and zooming.

This method updates the following attributes:

* xticks : the position of the ticks on the x axis
* yticks : the position of the ticks on the y axis
* xtext : the text of the ticks on the x axis
* ytext : the text of the ticks on the y axis

---

### phy.plot.BaseCanvas

Base canvas class. Derive from QOpenGLWindow.

The canvas represents an OpenGL-powered rectangular black window where one can add visuals
and attach interaction (pan/zoom, lasso) and layout (subplot) compaion objects.

---


**`BaseCanvas.add_visual(self, visual, **kwargs)`**

Add a visual to the canvas and build its OpenGL program using the attached interacts.

We can't build the visual's program before, because we need the canvas' transforms first.

**Parameters**


* `visual : Visual` 　 

* `clearable : True` 　 
    Whether the visual should be deleted when calling `canvas.clear()`.

* `exclude_origins : list-like` 　 
    List of interact instances that should not apply to that visual. For example, use to
    add a visual outside of the subplots, or with no support for pan and zoom.

* `key : str` 　 
    An optional key to identify a visual

---


**`BaseCanvas.attach_events(self, obj)`**

Attach an object that has `on_xxx()` methods. These methods are called when internal
events are raised by the canvas. This is used for mouse and key interactions.

---


**`BaseCanvas.clear(self)`**

Remove all visuals except those marked `clearable=False`.

---


**`BaseCanvas.emit(self, name, **kwargs)`**

Raise an internal event and call `on_xxx()` on attached objects.

---


**`BaseCanvas.event(self, e)`**

Touch event.

---


**`BaseCanvas.get_size(self)`**

Return the window size in pixels.

---


**`BaseCanvas.get_visual(self, key)`**

Get a visual from its key.

---


**`BaseCanvas.has_visual(self, visual)`**

Return whether a visual belongs to the canvas.

---


**`BaseCanvas.initializeGL(self)`**

Create the scene.

---


**`BaseCanvas.iter_update_queue(self)`**

Iterate through all OpenGL program updates called in lazy mode.

---


**`BaseCanvas.keyPressEvent(self, e)`**

Emit an internal `key_press` event.

---


**`BaseCanvas.keyReleaseEvent(self, e)`**

Emit an internal `key_release` event.

---


**`BaseCanvas.mouseDoubleClickEvent(self, e)`**

Emit an internal `mouse_double_click` event.

---


**`BaseCanvas.mouseMoveEvent(self, e)`**

Emit an internal `mouse_move` event.

---


**`BaseCanvas.mousePressEvent(self, e)`**

Emit an internal `mouse_press` event.

---


**`BaseCanvas.mouseReleaseEvent(self, e)`**

Emit an internal `mouse_release` or `mouse_click` event.

---


**`BaseCanvas.on_next_paint(self, f)`**

Register a function to be called at the next frame refresh (in paintGL()).

---


**`BaseCanvas.paintGL(self)`**

Draw all visuals.

---


**`BaseCanvas.remove(self, *visuals)`**

Remove some visuals objects from the canvas.

---


**`BaseCanvas.resizeEvent(self, e)`**

Emit a `resize(width, height)` event when resizing the window.

---


**`BaseCanvas.set_lazy(self, lazy)`**

When the lazy mode is enabled, all OpenGL calls are deferred. Use with
multithreading.

Must be called *after* the visuals have been added, but *before* set_data().

---


**`BaseCanvas.update(self)`**

Update the OpenGL canvas.

---


**`BaseCanvas.wheelEvent(self, e)`**

Emit an internal `mouse_wheel` event.

---


**`BaseCanvas.window_to_ndc(self, mouse_pos)`**

Convert a mouse position in pixels into normalized device coordinates, taking into
account pan and zoom.

---

### phy.plot.BaseLayout

Implement global transforms on a canvas, like subplots.

---


**`BaseLayout.attach(self, canvas)`**

Attach this layout to a canvas.

---


**`BaseLayout.box_map(self, mouse_pos)`**

Get the box and local NDC coordinates from mouse position.

---


**`BaseLayout.get_closest_box(self, ndc)`**

Override to return the box closest to a given position in NDC.

---


**`BaseLayout.imap(self, arr, box=None)`**

Inverse transformation from NDC to data coordinates.

---


**`BaseLayout.map(self, arr, box=None)`**

Direct transformation from data to NDC coordinates.

---


**`BaseLayout.update(self)`**

Update all visuals in the attached canvas.

---


**`BaseLayout.update_visual(self, visual)`**

Called whenever visual.set_data() is called. Set a_box_index in here.

---

### phy.plot.BaseVisual

A Visual represents one object (or homogeneous set of objects).

It is rendered with a single pass of a single gloo program with a single type of GL primitive.

**Main abstract methods**

validate
    takes as input the visual's parameters, set the default values, and validates all
    values
vertex_count
    takes as input the visual's parameters, and return the total number of vertices
set_data
    takes as input the visual's parameters, and ends with update calls to the underlying
    OpenGL program: `self.program[name] = data`

**Notes**

* set_data MUST set self.n_vertices (necessary for a_box_index in layouts)
* set_data MUST call `self.emit_visual_set_data()` at the end, and return the data

---


**`BaseVisual.add_batch_data(self, **kwargs)`**

Prepare data to be added later with `PlotCanvas.add_visual()`.

---


**`BaseVisual.close(self)`**

Close the visual.

---


**`BaseVisual.emit_visual_set_data(self)`**

Emit canvas.visual_set_data event after data has been set in the visual.

---


**`BaseVisual.hide(self)`**

Hide the visual.

---


**`BaseVisual.on_draw(self)`**

Draw the visual.

---


**`BaseVisual.on_resize(self, width, height)`**

Update the window size in the OpenGL program.

---


**`BaseVisual.reset_batch(self)`**

Reinitialize the batch.

---


**`BaseVisual.set_box_index(self, box_index, data=None)`**

Set the visual's box index. This is used by layouts (e.g. subplot indices).

---


**`BaseVisual.set_data(self)`**

Set data to the program.

Must be called *after* attach(canvas), because the program is built
when the visual is attached to the canvas.

---


**`BaseVisual.set_primitive_type(self, primitive_type)`**

Set the primitive type (points, lines, line_strip, line_fan, triangles).

---


**`BaseVisual.set_shader(self, name)`**

Set the built-in vertex and fragment shader.

---


**`BaseVisual.show(self)`**

Show the visual.

---


**`BaseVisual.validate(**kwargs)`**

Make consistent the input data for the visual.

---


**`BaseVisual.vertex_count(**kwargs)`**

Return the number of vertices as a function of the input data.

---

### phy.plot.BatchAccumulator

Accumulate data arrays for batch visuals.

This class is used to simplify the creation of batch visuals, where different visual elements
of the same type are concatenated into a singual Visual instance, which significantly
improves the performance of OpenGL.

---


**`BatchAccumulator.add(self, b, noconcat=(), n_items=None, n_vertices=None, **kwargs)`**

Add data for a given batch iteration.

**Parameters**


* `b : Bunch` 　 
    Data to add to the current batch iteration.

* `noconcat : tuple` 　 
    List of keys that should not be concatenated.

* `n_items : int` 　 
    Number of visual items to add in this batch iteration.

* `n_vertices : int` 　 
    Number of vertices added in this batch iteration.

**Note**

`n_items` and `n_vertices` differ for special visuals, like `TextVisual` where each
item is a string, but is represented in OpenGL as a number of vertices (six times the
number of characters, as each character requires two triangles).

---


**`BatchAccumulator.reset(self)`**

Reset the accumulator.

---


**`BatchAccumulator.data`**

Return the concatenated data as a dictionary.

---

### phy.plot.Boxed

Layout showing plots in rectangles at arbitrary positions. Used by the waveform view.

The boxes can be specified from their corner coordinates, or from their centers and
optional sizes. If the sizes are not specified, they will be computed automatically.
An iterative algorithm is used to find the largest box size that will not make them overlap.

**Constructor**


* `box_bounds : array-like` 　 
    A (n, 4) array where each row contains the `(xmin, ymin, xmax, ymax)`
    bounds of every box, in normalized device coordinates.

    Note: the box bounds need to be contained within [-1, 1] at all times,
    otherwise an error will be raised. This is to prevent silent clipping
    of the values when they are passed to a gloo Texture2D.


* `box_pos : array-like (2D, shape[1] == 2)` 　 
    Position of the centers of the boxes.

* `box_size : array-like (2D, shape[1] == 2)` 　 
    Size of the boxes.


* `box_var : str` 　 
    Name of the GLSL variable with the box index.

* `keep_aspect_ratio : boolean` 　 
    Whether to keep the aspect ratio of the bounds.

**Note**

To be used in a boxed layout, a visual must define `a_box_index` (by default) or another GLSL
variable specified in `box_var`.

---


**`Boxed.add_boxes(self, canvas)`**

Show the boxes borders.

---


**`Boxed.attach(self, canvas)`**

Attach the boxed interact to a canvas.

---


**`Boxed.box_map(self, mouse_pos)`**

Get the box and local NDC coordinates from mouse position.

---


**`Boxed.get_closest_box(self, pos)`**

Get the box closest to some position.

---


**`Boxed.imap(self, arr, box=None)`**

Apply the boxed inverse transformation to a position array.

---


**`Boxed.map(self, arr, box=None)`**

Apply the boxed transformation to a position array.

---


**`Boxed.update(self)`**

Update all visuals in the attached canvas.

---


**`Boxed.update_boxes(self, box_pos, box_size)`**

Set the box bounds from specified box positions and sizes.

---


**`Boxed.update_visual(self, visual)`**

Update a visual.

---


**`Boxed.box_bounds`**

Bounds of the boxes.

---


**`Boxed.box_pos`**

Position of the box centers.

---


**`Boxed.box_size`**

Sizes of the boxes.

---


**`Boxed.n_boxes`**

Total number of boxes.

---

### phy.plot.GLSLInserter

Object used to insert GLSL snippets into shader code.

This class provides methods to specify the snippets to insert, and the
`insert_into_shaders()` method inserts them into a vertex and fragment shader.

---


**`GLSLInserter.add_transform_chain(self, tc)`**

Insert all GLSL snippets from a transform chain.

---


**`GLSLInserter.insert_frag(self, glsl, location=None, origin=None, index=None)`**

Insert a GLSL snippet into the fragment shader. See `insert_vert()`.

---


**`GLSLInserter.insert_into_shaders(self, vertex, fragment, exclude_origins=())`**

Insert all GLSL snippets in a vertex and fragment shaders.

**Parameters**


* `vertex : str` 　 
    GLSL code of the vertex shader

* `fragment : str` 　 
    GLSL code of the fragment shader

* `exclude_origins : list-like` 　 
    List of interact instances to exclude when inserting the shaders.

**Notes**

The vertex shader typicall contains `gl_Position = transform(data_var_name);`
which is automatically detected, and the GLSL transformations are inserted there.

Snippets can contain `{{ var }}` placeholders for the transformed variable name.

---


**`GLSLInserter.insert_vert(self, glsl, location='transforms', origin=None, index=None)`**

Insert a GLSL snippet into the vertex shader.

**Parameters**


* `glsl : str` 　 
    The GLSL code to insert.

* `location : str` 　 
    Where to insert the GLSL code. Can be:

    * `header`: declaration of GLSL variables
    * `before_transforms`: just before the transforms in the vertex shader
    * `transforms`: where the GPU transforms are applied in the vertex shader
    * `after_transforms`: just after the GPU transforms


* `origin : Interact` 　 
    The interact object that adds this GLSL snippet. Should be discared by
    visuals that are added with that interact object in `exclude_origins`.

* `index : int` 　 
    Index of the snippets list to insert the snippet.

---

### phy.plot.Grid

Layout showing subplots arranged in a 2D grid.

**Constructor**


* `shape : tuple or str` 　 
    Number of rows, cols in the grid.

* `shape_var : str` 　 
    Name of the GLSL uniform variable that holds the shape, when it is variable.

* `box_var : str` 　 
    Name of the GLSL variable with the box index.

* `has_clip : boolean` 　 
    Whether subplots should be clipped.

**Note**

To be used in a grid, a visual must define `a_box_index` (by default) or another GLSL
variable specified in `box_var`.

---


**`Grid.add_boxes(self, canvas, shape=None)`**

Show subplot boxes.

---


**`Grid.attach(self, canvas)`**

Attach the grid to a canvas.

---


**`Grid.box_map(self, mouse_pos)`**

Get the box and local NDC coordinates from mouse position.

---


**`Grid.get_closest_box(self, pos)`**

Get the box index (i, j) closest to a given position in NDC coordinates.

---


**`Grid.imap(self, arr, box=None)`**

Apply the subplot inverse transformation to a position array.

---


**`Grid.map(self, arr, box=None)`**

Apply the subplot transformation to a position array.

---


**`Grid.update(self)`**

Update all visuals in the attached canvas.

---


**`Grid.update_visual(self, visual)`**

Update a visual.

---


**`Grid.shape`**

Return the grid shape.

---

### phy.plot.HistogramVisual

A histogram visual.

**Parameters**


* `hist : array-like (1D), or list of 1D arrays, or 2D array` 　 

* `color : array-like (2D, shape[1] == 4)` 　 

* `ylim : array-like (1D)` 　 
    The maximum hist value in the viewport.

---


**`HistogramVisual.add_batch_data(self, **kwargs)`**

Prepare data to be added later with `PlotCanvas.add_visual()`.

---


**`HistogramVisual.close(self)`**

Close the visual.

---


**`HistogramVisual.emit_visual_set_data(self)`**

Emit canvas.visual_set_data event after data has been set in the visual.

---


**`HistogramVisual.hide(self)`**

Hide the visual.

---


**`HistogramVisual.on_draw(self)`**

Draw the visual.

---


**`HistogramVisual.on_resize(self, width, height)`**

Update the window size in the OpenGL program.

---


**`HistogramVisual.reset_batch(self)`**

Reinitialize the batch.

---


**`HistogramVisual.set_box_index(self, box_index, data=None)`**

Set the visual's box index. This is used by layouts (e.g. subplot indices).

---


**`HistogramVisual.set_data(self, *args, **kwargs)`**

Update the visual data.

---


**`HistogramVisual.set_primitive_type(self, primitive_type)`**

Set the primitive type (points, lines, line_strip, line_fan, triangles).

---


**`HistogramVisual.set_shader(self, name)`**

Set the built-in vertex and fragment shader.

---


**`HistogramVisual.show(self)`**

Show the visual.

---


**`HistogramVisual.validate(self, hist=None, color=None, ylim=None, **kwargs)`**

Validate the requested data before passing it to set_data().

---


**`HistogramVisual.vertex_count(self, hist, **kwargs)`**

Number of vertices for the requested data.

---

### phy.plot.ImageVisual

Display a 2D image.

**Parameters**

* `image : array-like (3D)` 　 

---


**`ImageVisual.add_batch_data(self, **kwargs)`**

Prepare data to be added later with `PlotCanvas.add_visual()`.

---


**`ImageVisual.close(self)`**

Close the visual.

---


**`ImageVisual.emit_visual_set_data(self)`**

Emit canvas.visual_set_data event after data has been set in the visual.

---


**`ImageVisual.hide(self)`**

Hide the visual.

---


**`ImageVisual.on_draw(self)`**

Draw the visual.

---


**`ImageVisual.on_resize(self, width, height)`**

Update the window size in the OpenGL program.

---


**`ImageVisual.reset_batch(self)`**

Reinitialize the batch.

---


**`ImageVisual.set_box_index(self, box_index, data=None)`**

Set the visual's box index. This is used by layouts (e.g. subplot indices).

---


**`ImageVisual.set_data(self, *args, **kwargs)`**

Update the visual data.

---


**`ImageVisual.set_primitive_type(self, primitive_type)`**

Set the primitive type (points, lines, line_strip, line_fan, triangles).

---


**`ImageVisual.set_shader(self, name)`**

Set the built-in vertex and fragment shader.

---


**`ImageVisual.show(self)`**

Show the visual.

---


**`ImageVisual.validate(self, image=None, **kwargs)`**

Validate the requested data before passing it to set_data().

---


**`ImageVisual.vertex_count(self, image=None, **kwargs)`**

Number of vertices for the requested data.

---

### phy.plot.Lasso

Draw a polygon with the mouse and find the points that belong to the inside of the
polygon.

---


**`Lasso.add(self, pos)`**

Add a point to the polygon.

---


**`Lasso.attach(self, canvas)`**

Attach the lasso to a canvas.

---


**`Lasso.clear(self)`**

Reset the lasso.

---


**`Lasso.create_lasso_visual(self)`**

Create the lasso visual.

---


**`Lasso.in_polygon(self, pos)`**

Return which points belong to the polygon.

---


**`Lasso.on_mouse_click(self, e)`**

Add a polygon point with ctrl+click.

---


**`Lasso.update_lasso_visual(self)`**

Update the lasso visual with the current polygon.

---


**`Lasso.count`**

Number of vertices in the polygon.

---


**`Lasso.polygon`**

Coordinates of the polygon vertices.

---

### phy.plot.LineVisual

Line segments.

**Parameters**

* `pos : array-like (2D)` 　 

* `color : array-like (2D, shape[1] == 4)` 　 

* `data_bounds : array-like (2D, shape[1] == 4)` 　 

---


**`LineVisual.add_batch_data(self, **kwargs)`**

Prepare data to be added later with `PlotCanvas.add_visual()`.

---


**`LineVisual.close(self)`**

Close the visual.

---


**`LineVisual.emit_visual_set_data(self)`**

Emit canvas.visual_set_data event after data has been set in the visual.

---


**`LineVisual.hide(self)`**

Hide the visual.

---


**`LineVisual.on_draw(self)`**

Draw the visual.

---


**`LineVisual.on_resize(self, width, height)`**

Update the window size in the OpenGL program.

---


**`LineVisual.reset_batch(self)`**

Reinitialize the batch.

---


**`LineVisual.set_box_index(self, box_index, data=None)`**

Set the visual's box index. This is used by layouts (e.g. subplot indices).

---


**`LineVisual.set_data(self, *args, **kwargs)`**

Update the visual data.

---


**`LineVisual.set_primitive_type(self, primitive_type)`**

Set the primitive type (points, lines, line_strip, line_fan, triangles).

---


**`LineVisual.set_shader(self, name)`**

Set the built-in vertex and fragment shader.

---


**`LineVisual.show(self)`**

Show the visual.

---


**`LineVisual.validate(self, pos=None, color=None, data_bounds=None, **kwargs)`**

Validate the requested data before passing it to set_data().

---


**`LineVisual.vertex_count(self, pos=None, **kwargs)`**

Number of vertices for the requested data.

---

### phy.plot.PanZoom

Pan and zoom interact. Support mouse and keyboard interactivity.

**Constructor**


* `aspect : float` 　 
    Aspect ratio to keep while panning and zooming.

* `pan : 2-tuple` 　 
    Initial pan.

* `zoom : 2-tuple` 　 
    Initial zoom.

* `zmin : float` 　 
    Minimum zoom allowed.

* `zmax : float` 　 
    Maximum zoom allowed.

* `xmin : float` 　 
    Minimum x allowed.

* `xmax : float` 　 
    Maximum x allowed.

* `ymin : float` 　 
    Minimum y allowed.

* `ymax : float` 　 
    Maximum y allowed.

* `constrain_bounds : 4-tuple` 　 
    Equivalent to (xmin, ymin, xmax, ymax).

* `pan_var_name : str` 　 
    Name of the pan GLSL variable name

* `zoom_var_name : str` 　 
    Name of the zoom GLSL variable name

* `enable_mouse_wheel : boolean` 　 
    Whether to enable the mouse wheel for zooming.

**Interactivity**

* Keyboard arrows for panning
* Keyboard + and - for zooming
* Mouse left button + drag for panning
* Mouse right button + drag for zooming
* Mouse wheel for zooming
* R and double-click for reset

**Example**

```python

# Create and attach the PanZoom interact.
pz = PanZoom()
pz.attach(canvas)

# Create a visual.
visual = MyVisual(...)
visual.set_data(...)

# Attach the visual to the canvas.
canvas = BaseCanvas()
visual.attach(canvas, 'PanZoom')

canvas.show()
```

---


**`PanZoom.attach(self, canvas)`**

Attach this interact to a canvas.

---


**`PanZoom.get_range(self)`**

Return the bounds currently visible.

---


**`PanZoom.imap(self, arr)`**

Apply the current panzoom inverse transformation to a position array.

---


**`PanZoom.map(self, arr)`**

Apply the current panzoom transformation to a position array.

---


**`PanZoom.on_key_press(self, e)`**

Pan and zoom with the keyboard.

---


**`PanZoom.on_mouse_double_click(self, e)`**

Reset the view by double clicking anywhere in the canvas.

---


**`PanZoom.on_mouse_move(self, e)`**

Pan and zoom with the mouse.

---


**`PanZoom.on_mouse_wheel(self, e)`**

Zoom with the mouse wheel.

---


**`PanZoom.on_resize(self, e)`**

Resize event.

---


**`PanZoom.pan_delta(self, d)`**

Pan the view by a given amount.

---


**`PanZoom.reset(self)`**

Reset the view.

---


**`PanZoom.set_constrain_bounds(self, bounds)`**



---


**`PanZoom.set_pan_zoom(self, pan=None, zoom=None)`**

Set at once the pan and zoom.

---


**`PanZoom.set_range(self, bounds, keep_aspect=False)`**

Zoom to fit a box.

---


**`PanZoom.update(self)`**

Update all visuals in the attached canvas.

---


**`PanZoom.update_visual(self, visual)`**

Update a visual with the current pan and zoom values.

---


**`PanZoom.window_to_ndc(self, pos)`**

Return the mouse coordinates in NDC, taking panzoom into account.

---


**`PanZoom.zoom_delta(self, d, p=(0.0, 0.0), c=1.0)`**

Zoom the view by a given amount.

---


**`PanZoom.aspect`**

Aspect (width/height).

---


**`PanZoom.pan`**

Pan translation.

---


**`PanZoom.size`**

Window size of the canvas.

---


**`PanZoom.xmax`**

Maximum x allowed for pan.

---


**`PanZoom.xmin`**

Minimum x allowed for pan.

---


**`PanZoom.ymax`**

Maximum y allowed for pan.

---


**`PanZoom.ymin`**

Minimum y allowed for pan.

---


**`PanZoom.zmax`**

Maximal zoom level.

---


**`PanZoom.zmin`**

Minimum zoom level.

---


**`PanZoom.zoom`**

Zoom level.

---

### phy.plot.PlotCanvas

Plotting canvas that supports different layouts, subplots, lasso, axes, panzoom.

---


**`PlotCanvas.add_visual(self, visual, *args, **kwargs)`**

Add a visual and possibly set some data directly.

**Parameters**


* `visual : Visual` 　 

* `clearable : True` 　 
    Whether the visual should be deleted when calling `canvas.clear()`.

* `exclude_origins : list-like` 　 
    List of interact instances that should not apply to that visual. For example, use to
    add a visual outside of the subplots, or with no support for pan and zoom.

* `key : str` 　 
    An optional key to identify a visual

---


**`PlotCanvas.attach_events(self, obj)`**

Attach an object that has `on_xxx()` methods. These methods are called when internal
events are raised by the canvas. This is used for mouse and key interactions.

---


**`PlotCanvas.clear(self)`**

Remove all visuals except those marked `clearable=False`.

---


**`PlotCanvas.emit(self, name, **kwargs)`**

Raise an internal event and call `on_xxx()` on attached objects.

---


**`PlotCanvas.enable_axes(self, data_bounds=None, show_x=True, show_y=True)`**

Show axes in the canvas.

---


**`PlotCanvas.enable_lasso(self)`**

Enable lasso in the canvas.

---


**`PlotCanvas.enable_panzoom(self)`**

Enable pan zoom in the canvas.

---


**`PlotCanvas.event(self, e)`**

Touch event.

---


**`PlotCanvas.get_size(self)`**

Return the window size in pixels.

---


**`PlotCanvas.get_visual(self, key)`**

Get a visual from its key.

---


**`PlotCanvas.has_visual(self, visual)`**

Return whether a visual belongs to the canvas.

---


**`PlotCanvas.hist(self, *args, **kwargs)`**

Add a standalone (no batch) histogram plot.

---


**`PlotCanvas.initializeGL(self)`**

Create the scene.

---


**`PlotCanvas.iter_update_queue(self)`**

Iterate through all OpenGL program updates called in lazy mode.

---


**`PlotCanvas.keyPressEvent(self, e)`**

Emit an internal `key_press` event.

---


**`PlotCanvas.keyReleaseEvent(self, e)`**

Emit an internal `key_release` event.

---


**`PlotCanvas.lines(self, *args, **kwargs)`**

Add a standalone (no batch) line plot.

---


**`PlotCanvas.mouseDoubleClickEvent(self, e)`**

Emit an internal `mouse_double_click` event.

---


**`PlotCanvas.mouseMoveEvent(self, e)`**

Emit an internal `mouse_move` event.

---


**`PlotCanvas.mousePressEvent(self, e)`**

Emit an internal `mouse_press` event.

---


**`PlotCanvas.mouseReleaseEvent(self, e)`**

Emit an internal `mouse_release` or `mouse_click` event.

---


**`PlotCanvas.on_next_paint(self, f)`**

Register a function to be called at the next frame refresh (in paintGL()).

---


**`PlotCanvas.paintGL(self)`**

Draw all visuals.

---


**`PlotCanvas.plot(self, *args, **kwargs)`**

Add a standalone (no batch) plot.

---


**`PlotCanvas.polygon(self, *args, **kwargs)`**

Add a standalone (no batch) polygon plot.

---


**`PlotCanvas.remove(self, *visuals)`**

Remove some visuals objects from the canvas.

---


**`PlotCanvas.resizeEvent(self, e)`**

Emit a `resize(width, height)` event when resizing the window.

---


**`PlotCanvas.scatter(self, *args, **kwargs)`**

Add a standalone (no batch) scatter plot.

---


**`PlotCanvas.set_layout(self, layout=None, shape=None, n_plots=None, origin=None, box_bounds=None, box_pos=None, box_size=None, has_clip=None)`**

Set the plot layout: grid, boxed, stacked, or None.

---


**`PlotCanvas.set_lazy(self, lazy)`**

When the lazy mode is enabled, all OpenGL calls are deferred. Use with
multithreading.

Must be called *after* the visuals have been added, but *before* set_data().

---


**`PlotCanvas.text(self, *args, **kwargs)`**

Add a standalone (no batch) text plot.

---


**`PlotCanvas.update(self)`**

Update the OpenGL canvas.

---


**`PlotCanvas.update_visual(self, visual, *args, **kwargs)`**

Set the data of a visual, standalone or at the end of a batch.

---


**`PlotCanvas.uplot(self, *args, **kwargs)`**

Add a standalone (no batch) uniform plot.

---


**`PlotCanvas.uscatter(self, *args, **kwargs)`**

Add a standalone (no batch) uniform scatter plot.

---


**`PlotCanvas.wheelEvent(self, e)`**

Emit an internal `mouse_wheel` event.

---


**`PlotCanvas.window_to_ndc(self, mouse_pos)`**

Convert a mouse position in pixels into normalized device coordinates, taking into
account pan and zoom.

---


**`PlotCanvas.canvas`**



---

### phy.plot.PlotVisual

Plot visual, with multiple line plots of various sizes and colors.

**Parameters**


* `x : array-like (1D), or list of 1D arrays for different plots` 　 

* `y : array-like (1D), or list of 1D arrays, for different plots` 　 

* `color : array-like (2D, shape[-1] == 4)` 　 

* `depth : array-like (1D)` 　 

* `data_bounds : array-like (2D, shape[1] == 4)` 　 

---


**`PlotVisual.add_batch_data(self, **kwargs)`**

Prepare data to be added later with `PlotCanvas.add_visual()`.

---


**`PlotVisual.close(self)`**

Close the visual.

---


**`PlotVisual.emit_visual_set_data(self)`**

Emit canvas.visual_set_data event after data has been set in the visual.

---


**`PlotVisual.hide(self)`**

Hide the visual.

---


**`PlotVisual.on_draw(self)`**

Draw the visual.

---


**`PlotVisual.on_resize(self, width, height)`**

Update the window size in the OpenGL program.

---


**`PlotVisual.reset_batch(self)`**

Reinitialize the batch.

---


**`PlotVisual.set_box_index(self, box_index, data=None)`**

Set the visual's box index. This is used by layouts (e.g. subplot indices).

---


**`PlotVisual.set_color(self, color)`**

Update the visual's color.

---


**`PlotVisual.set_data(self, *args, **kwargs)`**

Update the visual data.

---


**`PlotVisual.set_primitive_type(self, primitive_type)`**

Set the primitive type (points, lines, line_strip, line_fan, triangles).

---


**`PlotVisual.set_shader(self, name)`**

Set the built-in vertex and fragment shader.

---


**`PlotVisual.show(self)`**

Show the visual.

---


**`PlotVisual.validate(self, x=None, y=None, color=None, depth=None, data_bounds=None, **kwargs)`**

Validate the requested data before passing it to set_data().

---


**`PlotVisual.vertex_count(self, y=None, **kwargs)`**

Number of vertices for the requested data.

---

### phy.plot.PolygonVisual

Polygon.

**Parameters**

* `pos : array-like (2D)` 　 

* `data_bounds : array-like (2D, shape[1] == 4)` 　 

---


**`PolygonVisual.add_batch_data(self, **kwargs)`**

Prepare data to be added later with `PlotCanvas.add_visual()`.

---


**`PolygonVisual.close(self)`**

Close the visual.

---


**`PolygonVisual.emit_visual_set_data(self)`**

Emit canvas.visual_set_data event after data has been set in the visual.

---


**`PolygonVisual.hide(self)`**

Hide the visual.

---


**`PolygonVisual.on_draw(self)`**

Draw the visual.

---


**`PolygonVisual.on_resize(self, width, height)`**

Update the window size in the OpenGL program.

---


**`PolygonVisual.reset_batch(self)`**

Reinitialize the batch.

---


**`PolygonVisual.set_box_index(self, box_index, data=None)`**

Set the visual's box index. This is used by layouts (e.g. subplot indices).

---


**`PolygonVisual.set_data(self, *args, **kwargs)`**

Update the visual data.

---


**`PolygonVisual.set_primitive_type(self, primitive_type)`**

Set the primitive type (points, lines, line_strip, line_fan, triangles).

---


**`PolygonVisual.set_shader(self, name)`**

Set the built-in vertex and fragment shader.

---


**`PolygonVisual.show(self)`**

Show the visual.

---


**`PolygonVisual.validate(self, pos=None, data_bounds=None, **kwargs)`**

Validate the requested data before passing it to set_data().

---


**`PolygonVisual.vertex_count(self, pos=None, **kwargs)`**

Number of vertices for the requested data.

---

### phy.plot.Range

Linear transform from a source rectangle to a target rectangle.

**Constructor**


* `from_bounds : 4-tuple` 　 
    Bounds of the source rectangle.

* `to_bounds : 4-tuple` 　 
    Bounds of the target rectangle.

---


**`Range.apply(self, arr, from_bounds=None, to_bounds=None)`**

Apply the transform to a NumPy array.

---


**`Range.glsl(self, var)`**

Return a GLSL snippet that applies the transform to a given GLSL variable name.

---


**`Range.inverse(self)`**

Return the inverse Range instance.

---

### phy.plot.Scale

Scaling transform.

---


**`Scale.apply(self, arr, value=None)`**

Apply a scaling to a NumPy array.

---


**`Scale.glsl(self, var)`**

Return a GLSL snippet that applies the scaling to a given GLSL variable name.

---


**`Scale.inverse(self)`**

Return the inverse Scale instance.

---

### phy.plot.ScatterVisual

Scatter visual, displaying a fixed marker at various positions, colors, and marker sizes.

**Constructor**


* `marker : string (used for all points in the scatter visual)` 　 
    Default: disc. Can be one of: arrow, asterisk, chevron, clover, club, cross, diamond,
    disc, ellipse, hbar, heart, infinity, pin, ring, spade, square, tag, triangle, vbar

**Parameters**


* `x : array-like (1D)` 　 

* `y : array-like (1D)` 　 

* `pos : array-like (2D)` 　 

* `color : array-like (2D, shape[1] == 4)` 　 

* `size : array-like (1D)` 　 
    Marker sizes, in pixels

* `depth : array-like (1D)` 　 

* `data_bounds : array-like (2D, shape[1] == 4)` 　 

---


**`ScatterVisual.add_batch_data(self, **kwargs)`**

Prepare data to be added later with `PlotCanvas.add_visual()`.

---


**`ScatterVisual.close(self)`**

Close the visual.

---


**`ScatterVisual.emit_visual_set_data(self)`**

Emit canvas.visual_set_data event after data has been set in the visual.

---


**`ScatterVisual.hide(self)`**

Hide the visual.

---


**`ScatterVisual.on_draw(self)`**

Draw the visual.

---


**`ScatterVisual.on_resize(self, width, height)`**

Update the window size in the OpenGL program.

---


**`ScatterVisual.reset_batch(self)`**

Reinitialize the batch.

---


**`ScatterVisual.set_box_index(self, box_index, data=None)`**

Set the visual's box index. This is used by layouts (e.g. subplot indices).

---


**`ScatterVisual.set_color(self, color)`**

Change the color of the markers.

---


**`ScatterVisual.set_data(self, *args, **kwargs)`**

Update the visual data.

---


**`ScatterVisual.set_marker_size(self, marker_size)`**

Change the size of the markers.

---


**`ScatterVisual.set_primitive_type(self, primitive_type)`**

Set the primitive type (points, lines, line_strip, line_fan, triangles).

---


**`ScatterVisual.set_shader(self, name)`**

Set the built-in vertex and fragment shader.

---


**`ScatterVisual.show(self)`**

Show the visual.

---


**`ScatterVisual.validate(self, x=None, y=None, pos=None, color=None, size=None, depth=None, data_bounds=None, **kwargs)`**

Validate the requested data before passing it to set_data().

---


**`ScatterVisual.vertex_count(self, x=None, y=None, pos=None, **kwargs)`**

Number of vertices for the requested data.

---

### phy.plot.Subplot

Transform to a grid subplot rectangle.

**Constructor**


* `shape : 2-tuple` 　 
    Number of rows and columns in the grid.

* `index : 2-tuple` 　 
    Row and column index of the subplot to transform into.

---


**`Subplot.apply(self, arr, from_bounds=None, to_bounds=None)`**

Apply the transform to a NumPy array.

---


**`Subplot.glsl(self, var)`**

Return a GLSL snippet that applies the transform to a given GLSL variable name.

---


**`Subplot.inverse(self)`**

Return the inverse Range instance.

---

### phy.plot.TextVisual

Display strings at multiple locations.

**Constructor**


* `color : 4-tuple` 　 

**Parameters**


* `pos : array-like (2D)` 　 
    Position of each string (of variable length).

* `text : list of strings (variable lengths)` 　 

* `anchor : array-like (2D)` 　 
    For each string, specifies the anchor of the string with respect to the string's position.

    Examples:

    * (0, 0): text centered at the position
    * (1, 1): the position is at the lower left of the string
    * (1, -1): the position is at the upper left of the string
    * (-1, 1): the position is at the lower right of the string
    * (-1, -1): the position is at the upper right of the string

    Values higher than 1 or lower than -1 can be used as margins, knowing that the unit
    of the anchor is (string width, string height).


* `data_bounds : array-like (2D, shape[1] == 4)` 　 

---


**`TextVisual.add_batch_data(self, **kwargs)`**

Prepare data to be added later with `PlotCanvas.add_visual()`.

---


**`TextVisual.close(self)`**

Close the visual.

---


**`TextVisual.emit_visual_set_data(self)`**

Emit canvas.visual_set_data event after data has been set in the visual.

---


**`TextVisual.hide(self)`**

Hide the visual.

---


**`TextVisual.on_draw(self)`**

Draw the visual.

---


**`TextVisual.on_resize(self, width, height)`**

Update the window size in the OpenGL program.

---


**`TextVisual.reset_batch(self)`**

Reinitialize the batch.

---


**`TextVisual.set_box_index(self, box_index, data=None)`**

Set the visual's box index. This is used by layouts (e.g. subplot indices).

---


**`TextVisual.set_data(self, *args, **kwargs)`**

Update the visual data.

---


**`TextVisual.set_primitive_type(self, primitive_type)`**

Set the primitive type (points, lines, line_strip, line_fan, triangles).

---


**`TextVisual.set_shader(self, name)`**

Set the built-in vertex and fragment shader.

---


**`TextVisual.show(self)`**

Show the visual.

---


**`TextVisual.validate(self, pos=None, text=None, anchor=None, data_bounds=None, **kwargs)`**

Validate the requested data before passing it to set_data().

---


**`TextVisual.vertex_count(self, **kwargs)`**

Number of vertices for the requested data.

---

### phy.plot.TransformChain

A linear sequence of transforms that happen on the CPU and GPU.

---


**`TransformChain.add_on_cpu(self, transforms, origin=None)`**

Add some transforms on the CPU.

---


**`TransformChain.add_on_gpu(self, transforms, origin=None)`**

Add some transforms on the GPU.

---


**`TransformChain.apply(self, arr)`**

Apply all CPU transforms on an array.

---


**`TransformChain.get(self, class_name)`**

Get a transform in the chain from its name.

---


**`TransformChain.inverse(self)`**

Return the inverse chain of transforms.

---


**`TransformChain.cpu_transforms`**

List of CPU transforms.

---


**`TransformChain.gpu_transforms`**

List of GPU transforms.

---

### phy.plot.Translate

Translation transform.

---


**`Translate.apply(self, arr, value=None)`**

Apply a translation to a NumPy array.

---


**`Translate.glsl(self, var)`**

Return a GLSL snippet that applies the translation to a given GLSL variable name.

---


**`Translate.inverse(self)`**

Return the inverse Translate instance.

---

### phy.plot.UniformPlotVisual

A plot visual with a uniform color.

**Constructor**


* `color : 4-tuple` 　 

* `depth : scalar` 　 

**Parameters**


* `x : array-like (1D), or list of 1D arrays for different plots` 　 

* `y : array-like (1D), or list of 1D arrays, for different plots` 　 

* `masks : array-like (1D)` 　 
    Similar to an alpha channel, but for color saturation instead of transparency.

* `data_bounds : array-like (2D, shape[1] == 4)` 　 

---


**`UniformPlotVisual.add_batch_data(self, **kwargs)`**

Prepare data to be added later with `PlotCanvas.add_visual()`.

---


**`UniformPlotVisual.close(self)`**

Close the visual.

---


**`UniformPlotVisual.emit_visual_set_data(self)`**

Emit canvas.visual_set_data event after data has been set in the visual.

---


**`UniformPlotVisual.hide(self)`**

Hide the visual.

---


**`UniformPlotVisual.on_draw(self)`**

Draw the visual.

---


**`UniformPlotVisual.on_resize(self, width, height)`**

Update the window size in the OpenGL program.

---


**`UniformPlotVisual.reset_batch(self)`**

Reinitialize the batch.

---


**`UniformPlotVisual.set_box_index(self, box_index, data=None)`**

Set the visual's box index. This is used by layouts (e.g. subplot indices).

---


**`UniformPlotVisual.set_data(self, *args, **kwargs)`**

Update the visual data.

---


**`UniformPlotVisual.set_primitive_type(self, primitive_type)`**

Set the primitive type (points, lines, line_strip, line_fan, triangles).

---


**`UniformPlotVisual.set_shader(self, name)`**

Set the built-in vertex and fragment shader.

---


**`UniformPlotVisual.show(self)`**

Show the visual.

---


**`UniformPlotVisual.validate(self, x=None, y=None, masks=None, data_bounds=None, **kwargs)`**

Validate the requested data before passing it to set_data().

---


**`UniformPlotVisual.vertex_count(self, y=None, **kwargs)`**

Number of vertices for the requested data.

---

### phy.plot.UniformScatterVisual

Scatter visual with a fixed marker color and size.

**Constructor**


* `marker : str` 　 

* `color : 4-tuple` 　 

* `size : scalar` 　 

**Parameters**


* `x : array-like (1D)` 　 

* `y : array-like (1D)` 　 

* `pos : array-like (2D)` 　 

* `masks : array-like (1D)` 　 
    Similar to an alpha channel, but for color saturation instead of transparency.

* `data_bounds : array-like (2D, shape[1] == 4)` 　 

---


**`UniformScatterVisual.add_batch_data(self, **kwargs)`**

Prepare data to be added later with `PlotCanvas.add_visual()`.

---


**`UniformScatterVisual.close(self)`**

Close the visual.

---


**`UniformScatterVisual.emit_visual_set_data(self)`**

Emit canvas.visual_set_data event after data has been set in the visual.

---


**`UniformScatterVisual.hide(self)`**

Hide the visual.

---


**`UniformScatterVisual.on_draw(self)`**

Draw the visual.

---


**`UniformScatterVisual.on_resize(self, width, height)`**

Update the window size in the OpenGL program.

---


**`UniformScatterVisual.reset_batch(self)`**

Reinitialize the batch.

---


**`UniformScatterVisual.set_box_index(self, box_index, data=None)`**

Set the visual's box index. This is used by layouts (e.g. subplot indices).

---


**`UniformScatterVisual.set_data(self, *args, **kwargs)`**

Update the visual data.

---


**`UniformScatterVisual.set_primitive_type(self, primitive_type)`**

Set the primitive type (points, lines, line_strip, line_fan, triangles).

---


**`UniformScatterVisual.set_shader(self, name)`**

Set the built-in vertex and fragment shader.

---


**`UniformScatterVisual.show(self)`**

Show the visual.

---


**`UniformScatterVisual.validate(self, x=None, y=None, pos=None, masks=None, data_bounds=None, **kwargs)`**

Validate the requested data before passing it to set_data().

---


**`UniformScatterVisual.vertex_count(self, x=None, y=None, pos=None, **kwargs)`**

Number of vertices for the requested data.

---

## phy.cluster

Manual clustering facilities.

---


**`phy.cluster.select_traces(traces, interval, sample_rate=None)`**

Load traces in an interval (in seconds).

---

### phy.cluster.ClusterMeta

Handle cluster metadata changes.

---


**`ClusterMeta.add_field(self, name, default_value=None)`**

Add a field with an optional default value.

---


**`ClusterMeta.from_dict(self, dic)`**

Import data from a `{cluster_id: {field: value}}` dictionary.

---


**`ClusterMeta.get(self, field, cluster)`**

Retrieve the value of one cluster for a given field.

---


**`ClusterMeta.redo(self)`**

Redo the next metadata change.

**Returns**


* `up : UpdateInfo instance` 　 

---


**`ClusterMeta.set(self, field, clusters, value, add_to_stack=True)`**

Set the value of one of several clusters.

**Parameters**


* `field : str` 　 
    The field to set.

* `clusters : list` 　 
    The list of cluster ids to change.

* `value : str` 　 
    The new metadata value for the given clusters.

* `add_to_stack : boolean` 　 
    Whether this metadata change should be recorded in the undo stack.

**Returns**


* `up : UpdateInfo instance` 　 

---


**`ClusterMeta.set_from_descendants(self, descendants, largest_old_cluster=None)`**

Update metadata of some clusters given the metadata of their ascendants.

**Parameters**


* `descendants : list` 　 
    List of pairs (old_cluster_id, new_cluster_id)

* `largest_old_cluster : int` 　 
    If available, the cluster id of the largest old cluster, used as a reference.

---


**`ClusterMeta.to_dict(self, field)`**

Export data to a `{cluster_id: value}` dictionary, for a particular field.

---


**`ClusterMeta.undo(self)`**

Undo the last metadata change.

**Returns**


* `up : UpdateInfo instance` 　 

---


**`ClusterMeta.fields`**

List of fields.

---

### phy.cluster.ClusterView

Display a table of all clusters with metrics and labels as columns. Derive from Table.

**Constructor**


* `parent : Qt widget` 　 

* `data : list` 　 
    List of dictionaries mapping fields to values.

* `columns : list` 　 
    List of columns in the table.

* `sort : 2-tuple` 　 
    Initial sort of the table as a pair (column_name, order), where order is
    either `asc` or `desc`.

---


**`ClusterView.add(self, objects)`**

Add objects object to the table.

---


**`ClusterView.build(self, callback=None)`**

Rebuild the HTML code of the widget.

---


**`ClusterView.change(self, objects)`**

Change some objects.

---


**`ClusterView.eval_js(self, expr, callback=None)`**

Evaluate a Javascript expression.

The `table` Javascript variable can be used to interact with the underlying Javascript
table.

The table has sortable columns, a filter text box, support for single and multi selection
of rows. Rows can be skippable (used for ignored clusters in phy).

The table can raise Javascript events that are relayed to Python. Objects are
transparently serialized and deserialized in JSON. Basic types (numbers, strings, lists)
are transparently converted between Python and Javascript.

**Parameters**


* `expr : str` 　 
    A Javascript expression.

* `callback : function` 　 
    A Python function that is called once the Javascript expression has been
    evaluated. It takes as input the output of the Javascript expression.

---


**`ClusterView.filter(self, text='')`**

Filter the view with a Javascript expression.

---


**`ClusterView.first(self, callback=None)`**

Select the first item.

---


**`ClusterView.get(self, id, callback=None)`**

Get the object given its id.

---


**`ClusterView.get_current_sort(self, callback=None)`**

Get the current sort as a tuple `(name, dir)`.

---


**`ClusterView.get_ids(self, callback=None)`**

Get the list of ids.

---


**`ClusterView.get_next_id(self, callback=None)`**

Get the next non-skipped row id.

---


**`ClusterView.get_previous_id(self, callback=None)`**

Get the previous non-skipped row id.

---


**`ClusterView.get_selected(self, callback=None)`**

Get the currently selected rows.

---


**`ClusterView.get_state(self, callback=None)`**

Return the cluster view state, with the current sort.

---


**`ClusterView.next(self, callback=None)`**

Select the next non-skipped row.

---


**`ClusterView.previous(self, callback=None)`**

Select the previous non-skipped row.

---


**`ClusterView.remove(self, ids)`**

Remove some objects from their ids.

---


**`ClusterView.remove_all(self)`**

Remove all rows in the table.

---


**`ClusterView.remove_all_and_add(self, objects)`**

Remove all rows in the table and add new objects.

---


**`ClusterView.select(self, ids, callback=None, **kwargs)`**

Select some rows in the table from Python.

This function calls `table.select()` in Javascript, which raises a Javascript event
relayed to Python. This sequence of actions is the same when the user selects
rows directly in the HTML view.

---


**`ClusterView.set_busy(self, busy)`**

Set the busy state of the GUI.

---


**`ClusterView.set_html(self, html, callback=None)`**

Set the HTML code.

---


**`ClusterView.set_state(self, state)`**

Set the cluster view state, with a specified sort.

---


**`ClusterView.sort_by(self, name, sort_dir='asc')`**

Sort by a given variable.

---


**`ClusterView.view_source(self, callback=None)`**

View the HTML source of the widget.

---

### phy.cluster.Clustering

Handle cluster changes in a set of spikes.

**Constructor**


* `spike_clusters : array-like` 　 
    Spike-cluster assignments, giving the cluster id of every spike.

* `new_cluster_id : int` 　 
    Cluster id that is not used yet (and not used in the cache if there is one). We need to
    ensure that cluster ids are unique and not reused in a given session.

* `spikes_per_cluster : dict` 　 
    Dictionary mapping each cluster id to the spike ids belonging to it. This is recomputed
    if not given. This object may take a while to compute, so it may be cached and passed
    to the constructor.

**Features**

* List of clusters appearing in a `spike_clusters` array
* Dictionary of spikes per cluster
* Merge
* Split and assign
* Undo/redo stack

**Notes**

The undo stack works by keeping the list of all spike cluster changes
made successively. Undoing consists of reapplying all changes from the
original `spike_clusters` array, except the last one.

**UpdateInfo**

Most methods of this class return an `UpdateInfo` instance. This object
contains information about the clustering changes done by the operation.
This object is used throughout the `phy.cluster.manual` package to let
different classes know about clustering changes.

`UpdateInfo` is a dictionary that also supports dot access (`Bunch` class).

---


**`Clustering.assign(self, spike_ids, spike_clusters_rel=0)`**

Make new spike cluster assignments.

**Parameters**


* `spike_ids : array-like` 　 
    List of spike ids.

* `spike_clusters_rel : array-like` 　 
    Relative cluster ids of the spikes in `spike_ids`. This
    must have the same size as `spike_ids`.

**Returns**


* `up : UpdateInfo instance` 　 

**Note**

`spike_clusters_rel` contain *relative* cluster indices. Their values
don't matter: what matters is whether two give spikes
should end up in the same cluster or not. Adding a constant number
to all elements in `spike_clusters_rel` results in exactly the same
operation.

The final cluster ids are automatically generated by the `Clustering`
class. This is because we must ensure that all modified clusters
get brand new ids. The whole library is based on the assumption that
cluster ids are unique and "disposable". Changing a cluster always
results in a new cluster id being assigned.

If a spike is assigned to a new cluster, then all other spikes
belonging to the same cluster are assigned to a brand new cluster,
even if they were not changed explicitely by the `assign()` method.

In other words, the list of spikes affected by an `assign()` is almost
always a strict superset of the `spike_ids` parameter. The only case
where this is not true is when whole clusters change: this is called
a merge. It is implemented in a separate `merge()` method because it
is logically much simpler, and faster to execute.

---


**`Clustering.merge(self, cluster_ids, to=None)`**

Merge several clusters to a new cluster.

**Parameters**


* `cluster_ids : array-like` 　 
    List of clusters to merge.

* `to : integer` 　 
    The id of the new cluster. By default, this is `new_cluster_id()`.

**Returns**


* `up : UpdateInfo instance` 　 

---


**`Clustering.new_cluster_id(self)`**

Generate a brand new cluster id.

**Note**

This new id strictly increases after an undo + new action,
meaning that old cluster ids are *not* reused. This ensures that
any cluster_id-based cache will always be valid even after undo
operations (i.e. no need for explicit cache invalidation in this case).

---


**`Clustering.redo(self)`**

Redo the last cluster assignment operation.

**Returns**


* `up : UpdateInfo instance of the changes done by this operation.` 　 

---


**`Clustering.reset(self)`**

Reset the clustering to the original clustering.

All changes are lost.

---


**`Clustering.spikes_in_clusters(self, clusters)`**

Return the array of spike ids belonging to a list of clusters.

---


**`Clustering.split(self, spike_ids, spike_clusters_rel=0)`**

Split a number of spikes into a new cluster.

This is equivalent to an `assign()` to a single new cluster.

**Parameters**


* `spike_ids : array-like` 　 
    Array of spike ids to split.

* `spike_clusters_rel : array-like (or None)` 　 
    Array of relative spike clusters.

**Returns**


* `up : UpdateInfo instance` 　 

**Note**

The note in the `assign()` method applies here as well. The list
of spikes affected by the split is almost always a strict superset
of the spike_ids parameter.

---


**`Clustering.undo(self)`**

Undo the last cluster assignment operation.

**Returns**


* `up : UpdateInfo instance of the changes done by this operation.` 　 

---


**`Clustering.cluster_ids`**

Ordered list of ids of all non-empty clusters.

---


**`Clustering.n_clusters`**

Total number of clusters.

---


**`Clustering.n_spikes`**

Number of spikes.

---


**`Clustering.spike_clusters`**

A n_spikes-long vector containing the cluster ids of all spikes.

---


**`Clustering.spike_ids`**

Array of all spike ids.

---


**`Clustering.spikes_per_cluster`**

A dictionary {cluster_id: spike_ids}.

---

### phy.cluster.CorrelogramView

A view showing the autocorrelogram of the selected clusters, and all cross-correlograms
of cluster pairs.

**Constructor**


* `correlograms : function` 　 
    Maps `(cluster_ids, bin_size, window_size)` to an `(n_clusters, n_clusters, n_bins) array`.


* `firing_rate : function` 　 
    Maps `(cluster_ids, bin_size)` to an `(n_clusters, n_clusters) array`

---


**`CorrelogramView.attach(self, gui)`**

Attach the view to the GUI.

---


**`CorrelogramView.close(self)`**

Close the underlying canvas.

---


**`CorrelogramView.decrease(self)`**

Decrease the window size.

---


**`CorrelogramView.increase(self)`**

Increase the window size.

---


**`CorrelogramView.on_mouse_wheel(self, e)`**

Change the scaling with the wheel.

---


**`CorrelogramView.on_select(self, cluster_ids=(), **kwargs)`**

Show the correlograms of the selected clusters.

---


**`CorrelogramView.screenshot(self, dir=None)`**

Save a PNG screenshot of the view into a given directory. By default, the screenshots
are saved in `~/.phy/screenshots/`.

---


**`CorrelogramView.set_bin(self, bin_size)`**

Set the correlogram bin size (in milliseconds).

Example: `1`

---


**`CorrelogramView.set_refractory_period(self, value)`**

Set the refractory period (in milliseconds).

---


**`CorrelogramView.set_state(self, state)`**

Set the view state.

The passed object is the persisted `self.state` bunch.

May be overriden.

---


**`CorrelogramView.set_status(self, message=None)`**

Set the status bar message in the GUI.

---


**`CorrelogramView.set_window(self, window_size)`**

Set the correlogram window size (in milliseconds).

Example: `100`

---


**`CorrelogramView.show(self)`**

Show the underlying canvas.

---


**`CorrelogramView.toggle_auto_update(self, checked)`**

When on, the view is automatically updated when the cluster selection changes.

---


**`CorrelogramView.toggle_normalization(self, checked)`**

Change the normalization of the correlograms.

---


**`CorrelogramView.state`**

View state, a Bunch instance automatically persisted in the GUI state when the
GUI is closed. To be overriden.

---

### phy.cluster.FeatureView

This view displays a 4x4 subplot matrix with different projections of the principal
component features. This view keeps track of which channels are currently shown.

**Constructor**


* `features : function` 　 
    Maps `(cluster_id, channel_ids=None, load_all=False)` to
    `Bunch(data, channel_ids, spike_ids , masks)`.
    * `data` is an `(n_spikes, n_channels, n_features)` array
    * `channel_ids` contains the channel ids of every row in `data`

    This allows for a sparse format.


* `attributes : dict` 　 
    Maps an attribute name to a 1D array with `n_spikes` numbers (for example, spike times).

---


**`FeatureView.attach(self, gui)`**

Attach the view to the GUI.

---


**`FeatureView.clear_channels(self)`**

Reset the current channels.

---


**`FeatureView.close(self)`**

Close the underlying canvas.

---


**`FeatureView.decrease(self)`**

Decrease the scaling of the features.

---


**`FeatureView.decrease_marker(self)`**

Decrease the marker size.

---


**`FeatureView.increase(self)`**

Increase the scaling of the features.

---


**`FeatureView.increase_marker(self)`**

Increase the marker size.

---


**`FeatureView.on_channel_click(self, sender=None, channel_id=None, key=None, button=None)`**

Respond to the click on a channel from another view, and update the
relevant subplots.

---


**`FeatureView.on_mouse_wheel(self, e)`**

Change the scaling with the wheel.

---


**`FeatureView.on_request_split(self, sender=None)`**

Return the spikes enclosed by the lasso.

---


**`FeatureView.on_select(self, cluster_ids=(), **kwargs)`**

Update the view with the selected clusters.

---


**`FeatureView.screenshot(self, dir=None)`**

Save a PNG screenshot of the view into a given directory. By default, the screenshots
are saved in `~/.phy/screenshots/`.

---


**`FeatureView.set_grid_dim(self, grid_dim)`**

Change the grid dim dynamically.

---


**`FeatureView.set_state(self, state)`**

Set the view state.

The passed object is the persisted `self.state` bunch.

May be overriden.

---


**`FeatureView.set_status(self, message=None)`**

Set the status bar message in the GUI.

---


**`FeatureView.show(self)`**

Show the underlying canvas.

---


**`FeatureView.toggle_auto_update(self, checked)`**

When on, the view is automatically updated when the cluster selection changes.

---


**`FeatureView.toggle_automatic_channel_selection(self, checked)`**

Toggle the automatic selection of channels when the cluster selection changes.

---


**`FeatureView.marker_size`**

Size of the spike markers, in pixels.

---


**`FeatureView.scaling`**

Scaling of the features.

---


**`FeatureView.state`**

View state, a Bunch instance automatically persisted in the GUI state when the
GUI is closed. To be overriden.

---

### phy.cluster.HistogramView

This view displays a histogram for every selected cluster, along with a possible plot
and some text. To be overriden.

**Constructor**


* `cluster_stat : function` 　 
    Maps `cluster_id` to `Bunch(histogram (1D array), plot (1D array), text)`.

---


**`HistogramView.attach(self, gui)`**

Attach the view to the GUI.

---


**`HistogramView.close(self)`**

Close the underlying canvas.

---


**`HistogramView.decrease(self)`**

Decrease the histogram range on the x avis.

---


**`HistogramView.increase(self)`**

Increase the histogram range on the x avis.

---


**`HistogramView.on_mouse_wheel(self, e)`**

Change the scaling with the wheel.

---


**`HistogramView.on_select(self, cluster_ids=(), **kwargs)`**

Update the view with the selected clusters.

---


**`HistogramView.screenshot(self, dir=None)`**

Save a PNG screenshot of the view into a given directory. By default, the screenshots
are saved in `~/.phy/screenshots/`.

---


**`HistogramView.set_n_bins(self, n_bins)`**

Set the number of bins in the histogram.

---


**`HistogramView.set_state(self, state)`**

Set the view state.

The passed object is the persisted `self.state` bunch.

May be overriden.

---


**`HistogramView.set_status(self, message=None)`**

Set the status bar message in the GUI.

---


**`HistogramView.set_x_max(self, x_max)`**

Set the maximum value on the x axis for the histogram.

---


**`HistogramView.show(self)`**

Show the underlying canvas.

---


**`HistogramView.toggle_auto_update(self, checked)`**

When on, the view is automatically updated when the cluster selection changes.

---


**`HistogramView.state`**

View state, a Bunch instance automatically persisted in the GUI state when the
GUI is closed. To be overriden.

---

### phy.cluster.ManualClusteringView

Base class for clustering views.

Typical property objects:

- `self.canvas`: a `PlotCanvas` instance by default (can also be a `PlotCanvasMpl` instance).
- `self.default_shortcuts`: a dictionary with the default keyboard shortcuts for the view
- `self.shortcuts`: a dictionary with the actual keyboard shortcuts for the view (can be passed
  to the view's constructor).
- `self.state_attrs`: a tuple with all attributes that should be automatically saved in the
  view's global GUI state.
- `self.local_state_attrs`: like above, but for the local GUI state (dataset-dependent).

---


**`ManualClusteringView.attach(self, gui)`**

Attach the view to the GUI.

Perform the following:

- Add the view to the GUI.
- Update the view's attribute from the GUI state
- Add the default view actions (auto_update, screenshot)
- Bind the on_select() method to the select event raised by the supervisor.
  This runs on a background thread not to block the GUI thread.

---


**`ManualClusteringView.close(self)`**

Close the underlying canvas.

---


**`ManualClusteringView.on_mouse_wheel(self, e)`**

Change the scaling with the wheel.

---


**`ManualClusteringView.on_select(self, cluster_ids=None, **kwargs)`**

Callback functions when clusters are selected. To be overriden.

---


**`ManualClusteringView.screenshot(self, dir=None)`**

Save a PNG screenshot of the view into a given directory. By default, the screenshots
are saved in `~/.phy/screenshots/`.

---


**`ManualClusteringView.set_state(self, state)`**

Set the view state.

The passed object is the persisted `self.state` bunch.

May be overriden.

---


**`ManualClusteringView.set_status(self, message=None)`**

Set the status bar message in the GUI.

---


**`ManualClusteringView.show(self)`**

Show the underlying canvas.

---


**`ManualClusteringView.toggle_auto_update(self, checked)`**

When on, the view is automatically updated when the cluster selection changes.

---


**`ManualClusteringView.state`**

View state, a Bunch instance automatically persisted in the GUI state when the
GUI is closed. To be overriden.

---

### phy.cluster.ProbeView

This view displays the positions of all channels on the probe, highlighting channels
where the selected clusters belong.

**Constructor**


* `positions : array-like` 　 
    An `(n_channels, 2)` array with the channel positions

* `best_channels : function` 　 
    Maps `cluster_id` to the list of the best_channel_ids.

---


**`ProbeView.attach(self, gui)`**

Attach the view to the GUI.

Perform the following:

- Add the view to the GUI.
- Update the view's attribute from the GUI state
- Add the default view actions (auto_update, screenshot)
- Bind the on_select() method to the select event raised by the supervisor.
  This runs on a background thread not to block the GUI thread.

---


**`ProbeView.close(self)`**

Close the underlying canvas.

---


**`ProbeView.on_mouse_wheel(self, e)`**

Change the scaling with the wheel.

---


**`ProbeView.on_select(self, cluster_ids=(), **kwargs)`**

Update the view with the selected clusters.

---


**`ProbeView.screenshot(self, dir=None)`**

Save a PNG screenshot of the view into a given directory. By default, the screenshots
are saved in `~/.phy/screenshots/`.

---


**`ProbeView.set_state(self, state)`**

Set the view state.

The passed object is the persisted `self.state` bunch.

May be overriden.

---


**`ProbeView.set_status(self, message=None)`**

Set the status bar message in the GUI.

---


**`ProbeView.show(self)`**

Show the underlying canvas.

---


**`ProbeView.toggle_auto_update(self, checked)`**

When on, the view is automatically updated when the cluster selection changes.

---


**`ProbeView.state`**

View state, a Bunch instance automatically persisted in the GUI state when the
GUI is closed. To be overriden.

---

### phy.cluster.RasterView

This view shows a raster plot of all clusters.

**Constructor**


* `spike_times : array-like` 　 
    An `(n_spikes,)` array with the spike times, in seconds.

* `spike_clusters : array-like` 　 
    An `(n_spikes,)` array with the spike-cluster assignments.

* `cluster_ids : array-like` 　 
    The list of all clusters to show initially.

* `cluster_color_selector : ClusterColorSelector` 　 
    The object managing the color mapping.

---


**`RasterView.attach(self, gui)`**

Attach the view to the GUI.

---


**`RasterView.close(self)`**

Close the underlying canvas.

---


**`RasterView.decrease(self)`**

Decrease the marker size.

---


**`RasterView.increase(self)`**

Increase the marker size.

---


**`RasterView.on_mouse_click(self, e)`**

Select a cluster by clicking in the raster plot.

---


**`RasterView.on_mouse_wheel(self, e)`**

Change the scaling with the wheel.

---


**`RasterView.on_select(self, cluster_ids=(), **kwargs)`**

Update the view with the selected clusters.

---


**`RasterView.plot(self)`**

Make the raster plot.

---


**`RasterView.screenshot(self, dir=None)`**

Save a PNG screenshot of the view into a given directory. By default, the screenshots
are saved in `~/.phy/screenshots/`.

---


**`RasterView.set_cluster_ids(self, cluster_ids)`**

Set the shown clusters, which can be filtered and in any order (from top to bottom).

---


**`RasterView.set_spike_clusters(self, spike_clusters)`**

Set the spike clusters for all spikes.

---


**`RasterView.set_state(self, state)`**

Set the view state.

The passed object is the persisted `self.state` bunch.

May be overriden.

---


**`RasterView.set_status(self, message=None)`**

Set the status bar message in the GUI.

---


**`RasterView.show(self)`**

Show the underlying canvas.

---


**`RasterView.toggle_auto_update(self, checked)`**

When on, the view is automatically updated when the cluster selection changes.

---


**`RasterView.update_cluster_sort(self, cluster_ids)`**

Update the order of all clusters.

---


**`RasterView.update_color(self, selected_clusters=None)`**

Update the color of the spikes, depending on the selected clustersd.

---


**`RasterView.data_bounds`**

Bounds of the raster plot view.

---


**`RasterView.marker_size`**

Size of the spike markers, in pixels.

---


**`RasterView.state`**

View state, a Bunch instance automatically persisted in the GUI state when the
GUI is closed. To be overriden.

---

### phy.cluster.ScatterView

This view displays a scatter plot for all selected clusters.

**Constructor**


* `coords : function` 　 
    Maps `cluster_ids` to a list `[Bunch(x, y), ...]` for each cluster.

---


**`ScatterView.attach(self, gui)`**

Attach the view to the GUI.

Perform the following:

- Add the view to the GUI.
- Update the view's attribute from the GUI state
- Add the default view actions (auto_update, screenshot)
- Bind the on_select() method to the select event raised by the supervisor.
  This runs on a background thread not to block the GUI thread.

---


**`ScatterView.close(self)`**

Close the underlying canvas.

---


**`ScatterView.decrease(self)`**

Decrease the marker size.

---


**`ScatterView.increase(self)`**

Increase the marker size.

---


**`ScatterView.on_mouse_wheel(self, e)`**

Change the scaling with the wheel.

---


**`ScatterView.on_request_split(self, sender=None)`**

Return the spikes enclosed by the lasso.

---


**`ScatterView.on_select(self, cluster_ids=(), **kwargs)`**

Update the view with the selected clusters.

---


**`ScatterView.screenshot(self, dir=None)`**

Save a PNG screenshot of the view into a given directory. By default, the screenshots
are saved in `~/.phy/screenshots/`.

---


**`ScatterView.set_state(self, state)`**

Set the view state.

The passed object is the persisted `self.state` bunch.

May be overriden.

---


**`ScatterView.set_status(self, message=None)`**

Set the status bar message in the GUI.

---


**`ScatterView.show(self)`**

Show the underlying canvas.

---


**`ScatterView.toggle_auto_update(self, checked)`**

When on, the view is automatically updated when the cluster selection changes.

---


**`ScatterView.marker_size`**

Size of the spike markers, in pixels.

---


**`ScatterView.state`**

View state, a Bunch instance automatically persisted in the GUI state when the
GUI is closed. To be overriden.

---

### phy.cluster.SimilarityView

Display a table of clusters with metrics and labels as columns, and an additional
similarity column.

This view displays clusters similar to the clusters currently selected
in the cluster view.

**Events**

* request_similar_clusters(cluster_id)

---


**`SimilarityView.add(self, objects)`**

Add objects object to the table.

---


**`SimilarityView.build(self, callback=None)`**

Rebuild the HTML code of the widget.

---


**`SimilarityView.change(self, objects)`**

Change some objects.

---


**`SimilarityView.eval_js(self, expr, callback=None)`**

Evaluate a Javascript expression.

The `table` Javascript variable can be used to interact with the underlying Javascript
table.

The table has sortable columns, a filter text box, support for single and multi selection
of rows. Rows can be skippable (used for ignored clusters in phy).

The table can raise Javascript events that are relayed to Python. Objects are
transparently serialized and deserialized in JSON. Basic types (numbers, strings, lists)
are transparently converted between Python and Javascript.

**Parameters**


* `expr : str` 　 
    A Javascript expression.

* `callback : function` 　 
    A Python function that is called once the Javascript expression has been
    evaluated. It takes as input the output of the Javascript expression.

---


**`SimilarityView.filter(self, text='')`**

Filter the view with a Javascript expression.

---


**`SimilarityView.first(self, callback=None)`**

Select the first item.

---


**`SimilarityView.get(self, id, callback=None)`**

Get the object given its id.

---


**`SimilarityView.get_current_sort(self, callback=None)`**

Get the current sort as a tuple `(name, dir)`.

---


**`SimilarityView.get_ids(self, callback=None)`**

Get the list of ids.

---


**`SimilarityView.get_next_id(self, callback=None)`**

Get the next non-skipped row id.

---


**`SimilarityView.get_previous_id(self, callback=None)`**

Get the previous non-skipped row id.

---


**`SimilarityView.get_selected(self, callback=None)`**

Get the currently selected rows.

---


**`SimilarityView.get_state(self, callback=None)`**

Return the cluster view state, with the current sort.

---


**`SimilarityView.next(self, callback=None)`**

Select the next non-skipped row.

---


**`SimilarityView.previous(self, callback=None)`**

Select the previous non-skipped row.

---


**`SimilarityView.remove(self, ids)`**

Remove some objects from their ids.

---


**`SimilarityView.remove_all(self)`**

Remove all rows in the table.

---


**`SimilarityView.remove_all_and_add(self, objects)`**

Remove all rows in the table and add new objects.

---


**`SimilarityView.reset(self, cluster_ids)`**

Recreate the similarity view, given the selected clusters in the cluster view.

---


**`SimilarityView.select(self, ids, callback=None, **kwargs)`**

Select some rows in the table from Python.

This function calls `table.select()` in Javascript, which raises a Javascript event
relayed to Python. This sequence of actions is the same when the user selects
rows directly in the HTML view.

---


**`SimilarityView.set_busy(self, busy)`**

Set the busy state of the GUI.

---


**`SimilarityView.set_html(self, html, callback=None)`**

Set the HTML code.

---


**`SimilarityView.set_selected_index_offset(self, n)`**

Set the index of the selected cluster, used for correct coloring in the similarity
view.

---


**`SimilarityView.set_state(self, state)`**

Set the cluster view state, with a specified sort.

---


**`SimilarityView.sort_by(self, name, sort_dir='asc')`**

Sort by a given variable.

---


**`SimilarityView.view_source(self, callback=None)`**

View the HTML source of the widget.

---

### phy.cluster.Supervisor

Component that brings manual clustering facilities to a GUI:

* `Clustering` instance: merge, split, undo, redo.
* `ClusterMeta` instance: change cluster metadata (e.g. group).
* Cluster selection.
* Many manual clustering-related actions, snippets, shortcuts, etc.
* Two HTML tables : `ClusterView` and `SimilarityView`.

**Constructor**


* `spike_clusters : array-like` 　 
    Spike-clusters assignments.

* `cluster_groups : dict` 　 
    Maps a cluster id to a group name (noise, mea, good, None for unsorted).

* `cluster_metrics : dict` 　 
    Maps a metric name to a function `cluster_id => value`

* `similarity : function` 　 
    Maps a cluster id to a list of pairs `[(similar_cluster_id, similarity), ...]`

* `new_cluster_id : function` 　 
    Function that takes no argument and returns a brand new cluster id (smallest cluster id
    not used in the cache).

* `sort : 2-tuple` 　 
    Initial sort as a pair `(column_name, order)` where `order` is either `asc` or `desc`

* `context : Context` 　 
    Handles the cache.

**Events**

When this component is attached to a GUI, the following events are emitted:

* `select(cluster_ids)`
    When clusters are selected in the cluster view or similarity view.
* `cluster(up)`
    When a clustering action occurs, changing the spike clusters assignment of the cluster
    metadata.
* `attach_gui(gui)`
    When the Supervisor instance is attached to the GUI.
* `request_split()`
    When the user requests to split (typically, a lasso has been drawn before).
* `error(msg)`
    When an error is raised.
* `color_mapping_changed()`
    When the color mapping changed.
* `save_clustering(spike_clusters, cluster_groups, *cluster_labels)`
    When the user wants to save the spike cluster assignments and the cluster metadata.

---


**`Supervisor.attach(self, gui)`**

Attach to the GUI.

---


**`Supervisor.block(self)`**

Block until there are no pending actions.

Only used in the automated testing suite.

---


**`Supervisor.change_color_field(self, color_field)`**

Change the color field (the name of the cluster view column used for the selected
colormap).

---


**`Supervisor.change_colormap(self, colormap)`**

Change the colormap.

---


**`Supervisor.filter(self, text)`**

Filter the clusters using a Javascript expression on the column names.

---


**`Supervisor.get_labels(self, field)`**

Return the labels of all clusters, for a given label name.

---


**`Supervisor.is_dirty(self)`**

Return whether there are any pending changes.

---


**`Supervisor.label(self, name, value, cluster_ids=None)`**

Assign a label to some clusters.

---


**`Supervisor.merge(self, cluster_ids=None, to=None)`**

Merge the selected clusters.

---


**`Supervisor.move(self, group, which)`**

Assign a cluster group to some clusters.

---


**`Supervisor.n_spikes(self, cluster_id)`**

Number of spikes in a given cluster.

---


**`Supervisor.next(self, callback=None)`**

Select the next cluster in the similarity view.

---


**`Supervisor.next_best(self, callback=None)`**

Select the next best cluster in the cluster view.

---


**`Supervisor.previous(self, callback=None)`**

Select the previous cluster in the similarity view.

---


**`Supervisor.previous_best(self, callback=None)`**

Select the previous best cluster in the cluster view.

---


**`Supervisor.redo(self)`**

Undo the last undone action.

---


**`Supervisor.reset_wizard(self, callback=None)`**

Reset the wizard.

---


**`Supervisor.save(self)`**

Save the manual clustering back to disk.

This method emits the `save_clustering(spike_clusters, groups, *labels)` event.
It is up to the caller to react to this event and save the data to disk.

---


**`Supervisor.select(self, *cluster_ids, callback=None)`**

Select a list of clusters.

---


**`Supervisor.sort(self, column, sort_dir='desc')`**

Sort the cluster view by a given column, in a given order (asc or desc).

---


**`Supervisor.split(self, spike_ids=None, spike_clusters_rel=0)`**

Make a new cluster out of the specified spikes.

---


**`Supervisor.toggle_categorical_colormap(self, checked)`**

Use a categorical or continuous colormap.

---


**`Supervisor.toggle_logarithmic_colormap(self, checked)`**

Use a logarithmic transform or not for the colormap.

---


**`Supervisor.undo(self)`**

Undo the last action.

---


**`Supervisor.cluster_info`**

The cluster view table as a list of per-cluster dictionaries.

---


**`Supervisor.fields`**

List of all cluster label names.

---


**`Supervisor.selected`**

Selected clusters in the cluster and similarity views.

---


**`Supervisor.selected_clusters`**

Selected clusters in the cluster view only.

---


**`Supervisor.selected_similar`**

Selected clusters in the similarity view only.

---


**`Supervisor.state`**

GUI state, with the cluster view and similarity view states.

---

### phy.cluster.TemplateView

This view shows all template waveforms of all clusters in a large grid of shape
`(n_channels, n_clusters)`.

**Constructor**


* `templates : function` 　 
    Maps `cluster_ids` to a list of `[Bunch(template, channel_ids)]` where `template` is
    an `(n_samples, n_channels)` array, and `channel_ids` specifies the channels of the
    `template` array (sparse format).

* `channel_ids : array-like` 　 
    The list of all channel ids.

* `cluster_ids : array-like` 　 
    The list of all clusters to show initially.

* `cluster_color_selector : ClusterColorSelector` 　 
    The object managing the color mapping.

---


**`TemplateView.attach(self, gui)`**

Attach the view to the GUI.

---


**`TemplateView.close(self)`**

Close the underlying canvas.

---


**`TemplateView.decrease(self)`**

Decrease the scaling of the template waveforms.

---


**`TemplateView.increase(self)`**

Increase the scaling of the template waveforms.

---


**`TemplateView.on_mouse_click(self, e)`**

Select a cluster by clicking on its template waveform.

---


**`TemplateView.on_mouse_wheel(self, e)`**

Change the scaling with the wheel.

---


**`TemplateView.on_select(self, cluster_ids=(), **kwargs)`**

Update the view with the selected clusters.

---


**`TemplateView.plot(self)`**

Make the template plot.

---


**`TemplateView.screenshot(self, dir=None)`**

Save a PNG screenshot of the view into a given directory. By default, the screenshots
are saved in `~/.phy/screenshots/`.

---


**`TemplateView.set_cluster_ids(self, cluster_ids)`**



---


**`TemplateView.set_state(self, state)`**

Set the view state.

The passed object is the persisted `self.state` bunch.

May be overriden.

---


**`TemplateView.set_status(self, message=None)`**

Set the status bar message in the GUI.

---


**`TemplateView.show(self)`**

Show the underlying canvas.

---


**`TemplateView.toggle_auto_update(self, checked)`**

When on, the view is automatically updated when the cluster selection changes.

---


**`TemplateView.update_cluster_sort(self, cluster_ids)`**

Update the order of the clusters.

---


**`TemplateView.update_color(self, selected_clusters=None)`**

Update the color of the clusters, taking the selected clusters into account.

---


**`TemplateView.scaling`**

Scaling of the template waveforms.

---


**`TemplateView.state`**

View state, a Bunch instance automatically persisted in the GUI state when the
GUI is closed. To be overriden.

---

### phy.cluster.TraceView

This view shows the raw traces along with spike waveforms.

**Constructor**


* `traces : function` 　 
    Maps a time interval `(t0, t1)` to a `Bunch(data, color, waveforms)` where
    * `data` is an `(n_samples, n_channels)` array
    * `waveforms` is a list of bunchs with the following attributes:
        * `data`
        * `color`
        * `channel_ids`
        * `start_time`
        * `spike_id`
        * `spike_cluster`


* `spike_times : function` 　 
    Teturns the list of relevant spike times.

* `sample_rate : float` 　 

* `duration : float` 　 

* `n_channels : int` 　 

* `channel_vertical_order : array-like` 　 
    Permutation of the channels.

---


**`TraceView.attach(self, gui)`**

Attach the view to the GUI.

---


**`TraceView.close(self)`**

Close the underlying canvas.

---


**`TraceView.decrease(self)`**

Decrease the scaling of the traces.

---


**`TraceView.go_left(self)`**

Go to left.

---


**`TraceView.go_right(self)`**

Go to right.

---


**`TraceView.go_to(self, time)`**

Go to a specific time (in seconds).

---


**`TraceView.go_to_next_spike(self)`**

Jump to the next spike from the first selected cluster.

---


**`TraceView.go_to_previous_spike(self)`**

Jump to the previous spike from the first selected cluster.

---


**`TraceView.increase(self)`**

Increase the scaling of the traces.

---


**`TraceView.narrow(self)`**

Decrease the interval size.

---


**`TraceView.on_mouse_click(self, e)`**

Select a cluster by clicking on a spike.

---


**`TraceView.on_mouse_wheel(self, e)`**

Change the scaling with the wheel.

---


**`TraceView.on_select(self, cluster_ids=None, **kwargs)`**

Update the view with the selected clusters.

---


**`TraceView.screenshot(self, dir=None)`**

Save a PNG screenshot of the view into a given directory. By default, the screenshots
are saved in `~/.phy/screenshots/`.

---


**`TraceView.set_interval(self, interval=None, change_status=True)`**

Display the traces and spikes in a given interval.

---


**`TraceView.set_state(self, state)`**

Set the view state.

The passed object is the persisted `self.state` bunch.

May be overriden.

---


**`TraceView.set_status(self, message=None)`**

Set the status bar message in the GUI.

---


**`TraceView.shift(self, delay)`**

Shift the interval by a given delay (in seconds).

---


**`TraceView.show(self)`**

Show the underlying canvas.

---


**`TraceView.toggle_auto_update(self, checked)`**

When on, the view is automatically updated when the cluster selection changes.

---


**`TraceView.toggle_highlighted_spikes(self, checked)`**

Toggle between showing all spikes or selected spikes.

---


**`TraceView.toggle_show_labels(self, checked)`**

Toggle the display of the channel ids.

---


**`TraceView.widen(self)`**

Increase the interval size.

---


**`TraceView.half_duration`**

Half of the duration of the current interval.

---


**`TraceView.interval`**

Interval as `(tmin, tmax)`.

---


**`TraceView.origin`**

Whether to show the channels from top to bottom (`top` option, the default), or from
bottom to top (`bottom`).

---


**`TraceView.scaling`**

Scaling of the traces.

---


**`TraceView.stacked`**



---


**`TraceView.state`**

View state, a Bunch instance automatically persisted in the GUI state when the
GUI is closed. To be overriden.

---


**`TraceView.time`**

Time at the center of the window.

---

### phy.cluster.UpdateInfo

Object created every time the dataset is modified via a clustering or cluster metadata
action. It is passed to event callbacks that react to these changes. Derive from Bunch.

**Parameters**


* `description : str` 　 
    Information about the update: merge, assign, or metadata_xxx for metadata changes

* `history : str` 　 
    undo, redo, or None

* `spike_ids : array-like` 　 
    All spike ids that were affected by the clustering action.

* `added : list` 　 
    List of new cluster ids.

* `deleted : list` 　 
    List of cluster ids that were deleted during the action. There are no modified clusters:
    every change triggers the deletion of and addition of clusters.

* `descendants : list` 　 
    List of pairs (old_cluster_id, new_cluster_id), used to track the history of
    the clusters.

* `metadata_changed : list` 　 
    List of cluster ids that had a change of metadata.

* `metadata_value : str` 　 
    The new metadata value for the affected change.

* `undo_state : Bunch` 　 
    Returned during an undo, it contains information about the undone action. This is used
    when redoing the undone action.

---


**`UpdateInfo.copy(self)`**

Return a new Bunch instance which is a copy of the current Bunch instance.

---

### phy.cluster.WaveformView

This view shows the waveforms of the selected clusters, on relevant channels,
following the probe geometry.

**Constructor**


* `waveforms : function` 　 
    Maps a cluster id to a Bunch with the following attributes:
    * `data` : a 3D array `(n_spikes, n_samples, n_channels_loc)`
    * `channel_ids` : the channel ids corresponding to the third dimension in `data
    * `channel_positions` : a 2D array with the coordinates of the channels on the probe
    * `masks` : a 2D array `(n_spikes, n_channels)` with the waveforms masks
    * `alpha` : the alpha transparency channel


* `channel_labels : array-like` 　 
    Labels of the channels.

---


**`WaveformView.attach(self, gui)`**

Attach the view to the GUI.

---


**`WaveformView.close(self)`**

Close the underlying canvas.

---


**`WaveformView.decrease(self)`**

Decrease the vertical scaling of the waveforms.

---


**`WaveformView.extend_horizontally(self)`**

Increase the horizontal scaling of the probe.

---


**`WaveformView.extend_vertically(self)`**

Increase the vertical scaling of the waveforms.

---


**`WaveformView.increase(self)`**

Increase the vertical scaling of the waveforms.

---


**`WaveformView.narrow(self)`**

Decrease the horizontal scaling of the waveforms.

---


**`WaveformView.on_mouse_click(self, e)`**

Select a channel by clicking on a box in the waveform view.

---


**`WaveformView.on_mouse_wheel(self, e)`**

Change the scaling with the wheel.

---


**`WaveformView.on_select(self, cluster_ids=(), **kwargs)`**

Update the view with the selected clusters.

---


**`WaveformView.screenshot(self, dir=None)`**

Save a PNG screenshot of the view into a given directory. By default, the screenshots
are saved in `~/.phy/screenshots/`.

---


**`WaveformView.set_state(self, state)`**

Set the view state.

The passed object is the persisted `self.state` bunch.

May be overriden.

---


**`WaveformView.set_status(self, message=None)`**

Set the status bar message in the GUI.

---


**`WaveformView.show(self)`**

Show the underlying canvas.

---


**`WaveformView.shrink_horizontally(self)`**

Decrease the horizontal scaling of the waveforms.

---


**`WaveformView.shrink_vertically(self)`**

Decrease the vertical scaling of the waveforms.

---


**`WaveformView.toggle_auto_update(self, checked)`**

When on, the view is automatically updated when the cluster selection changes.

---


**`WaveformView.toggle_show_labels(self, checked)`**

Whether to show the channel ids or not.

---


**`WaveformView.toggle_waveform_overlap(self, checked)`**

Toggle the overlap of the waveforms.

---


**`WaveformView.widen(self)`**

Increase the horizontal scaling of the waveforms.

---


**`WaveformView.box_scaling`**

Scaling of the channel boxes.

---


**`WaveformView.boxed`**

Layout instance.

---


**`WaveformView.overlap`**

Whether to overlap the waveforms belonging to different clusters.

---


**`WaveformView.probe_scaling`**

Scaling of the entire probe.

---


**`WaveformView.state`**

View state, a Bunch instance automatically persisted in the GUI state when the
GUI is closed. To be overriden.

---

## phy.apps.template

Template GUI.

---


**`phy.apps.template.template_describe(params_path)`**

Describe a template dataset.

---


**`phy.apps.template.template_gui(params_path)`**

Launch the Template GUI.

---

### phy.apps.template.TemplateController

Controller for the Template GUI.

**Constructor**

* `dat_path : str or Path or list` 　 
    Path to the raw data file(s)

* `config_dir : str or Path` 　 
    Path to the configuration directory

* `model : Model` 　 
    Model object, optional (it is automatically created otherwise)

* `plugins : list` 　 
    List of plugins to manually activate, optional (the plugins are automatically loaded from
    the user configuration directory).

---


**`TemplateController.create_amplitude_view(self)`**

Create an amplitude view.

---


**`TemplateController.create_correlogram_view(self)`**

Create a correlogram view.

---


**`TemplateController.create_feature_view(self)`**



---


**`TemplateController.create_gui(self, default_views=None, **kwargs)`**

Create the template GUI.

**Constructor**


* `default_views : list` 　 
    List of views to add in the GUI, optional. By default, all views from the view
    count are added.

---


**`TemplateController.create_ipython_view(self)`**

Create an IPython View.

---


**`TemplateController.create_probe_view(self)`**

Create a probe view.

---


**`TemplateController.create_raster_view(self)`**

Create a raster view.

---


**`TemplateController.create_template_feature_view(self)`**



---


**`TemplateController.create_template_view(self)`**

Create a template view.

---


**`TemplateController.create_trace_view(self)`**

Create a trace view.

---


**`TemplateController.create_waveform_view(self)`**



---


**`TemplateController.get_amplitude_histogram(self, cluster_id)`**

Return the spike amplitude data of a cluster.

---


**`TemplateController.get_amplitudes(self, cluster_ids, load_all=False)`**

Get the spike amplitudes for a set of clusters.

---


**`TemplateController.get_best_channel(self, cluster_id)`**

Return the best channel of a given cluster.

---


**`TemplateController.get_best_channels(self, cluster_id)`**

Return the best channels of a given cluster.

---


**`TemplateController.get_cluster_amplitude(self, cluster_id)`**

Get the template waveform amplitude of a cluster.

---


**`TemplateController.get_correlograms(self, cluster_ids, bin_size, window_size)`**

Return the cross- and auto-correlograms of a set of clusters.

---


**`TemplateController.get_correlograms_rate(self, cluster_ids, bin_size)`**

Return the baseline firing rate of the cross- and auto-correlograms of clusters.

---


**`TemplateController.get_features(self, cluster_id=None, channel_ids=None, load_all=None)`**

Return the features of a given cluster on specified channels.

---


**`TemplateController.get_firing_rate(self, cluster_id)`**

Return the firing rate data of a cluster.

---


**`TemplateController.get_isi(self, cluster_id)`**

Return the ISI data of a cluster.

---


**`TemplateController.get_mean_firing_rate(self, cluster_id)`**

Return the mean firing rate of a cluster.

---


**`TemplateController.get_mean_waveforms(self, cluster_id)`**

Get the mean waveform of a cluster on its best channels.

---


**`TemplateController.get_probe_depth(self, cluster_id)`**

Return the depth of a cluster.

---


**`TemplateController.get_spike_ids(self, cluster_id=None, load_all=None)`**

Return some or all spikes belonging to a given cluster.

---


**`TemplateController.get_spike_times(self, cluster_id=None, load_all=None)`**

Return the times of some or all spikes belonging to a given cluster.

---


**`TemplateController.get_template_counts(self, cluster_id)`**

Return a histogram of the number of spikes in each template for a given cluster.

---


**`TemplateController.get_template_features(self, cluster_ids, load_all=None)`**

Get the template features of a pair of clusters.

---


**`TemplateController.get_template_for_cluster(self, cluster_id)`**

Return the largest template associated to a cluster.

---


**`TemplateController.get_template_waveforms(self, cluster_id)`**

Return the waveforms of the templates corresponding to a cluster.

---


**`TemplateController.get_templates(self, cluster_ids)`**

Get the template waveforms of a set of clusters.

---


**`TemplateController.get_traces(self, interval, show_all_spikes=False)`**

Get traces and spike waveforms.

---


**`TemplateController.get_waveforms(self, cluster_id)`**

Return a selection of waveforms for a cluster.

---


**`TemplateController.similarity(self, cluster_id)`**

Return the list of similar clusters to a given cluster.

---

### phy.apps.template.TemplateModel

Object holding all data of a KiloSort/phy dataset.

**Constructor**


* `dat_path : str, Path, or list` 　 
    Path to the raw data files.

* `dir_path : str or Path` 　 
    Path to the dataset directory

* `dtype : NumPy dtype` 　 
    Data type of the raw data file

* `offset : int` 　 
    Header offset of the binary file

* `n_channels_dat : int` 　 
    Number of channels in the dat file

* `sample_rate : float` 　 
    Sampling rate of the data file.

* `filter_order : int` 　 
    Order of the filter used for waveforms

* `hp_filtered : bool` 　 
    Whether the raw data file is already high-pass filtered. In that case, disable the
    filtering for the waveform extraction.

---


**`TemplateModel.describe(self)`**

Display basic information about the dataset.

---


**`TemplateModel.get_cluster_channels(self, cluster_id)`**

Return the most relevant channels of a cluster.

---


**`TemplateModel.get_cluster_spike_waveforms(self, cluster_id)`**

Return all spike waveforms of a cluster, on the most relevant channels.

---


**`TemplateModel.get_cluster_spikes(self, cluster_id)`**

Return the spike ids that belong to a given template.

---


**`TemplateModel.get_features(self, spike_ids, channel_ids)`**

Return sparse features for given spikes.

---


**`TemplateModel.get_metadata(self, name)`**

Return a dictionary {cluster_id: value} for a cluster metadata
field.

---


**`TemplateModel.get_template(self, template_id, channel_ids=None)`**

Get data about a template.

---


**`TemplateModel.get_template_channels(self, template_id)`**

Return the most relevant channels of a template.

---


**`TemplateModel.get_template_features(self, spike_ids)`**

Return sparse template features for given spikes.

---


**`TemplateModel.get_template_spike_waveforms(self, template_id)`**

Return all spike waveforms of a template, on the most relevant channels.

---


**`TemplateModel.get_template_spikes(self, template_id)`**

Return the spike ids that belong to a given template.

---


**`TemplateModel.get_template_waveforms(self, template_id)`**

Return the waveforms of a template on the most relevant channels.

---


**`TemplateModel.get_waveforms(self, spike_ids, channel_ids)`**

Return spike waveforms on specified channels.

---


**`TemplateModel.save_mean_waveforms(self, mean_waveforms)`**

Save the mean waveforms as a single array.

---


**`TemplateModel.save_metadata(self, name, values)`**

Save a dictionary {cluster_id: value} with cluster metadata in
a TSV file.

---


**`TemplateModel.save_spike_clusters(self, spike_clusters)`**

Save the spike clusters.

---


**`TemplateModel.metadata_fields`**

List of metadata fields.

---
