
import copy

import wx
import wx.lib.scrolledpanel
import numpy
from keras import backend
from matplotlib.pyplot import figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

from main import DEFAULT_WIDTH, DEFAULT_HEIGHT, MODE
from main import TRAIN_DATASETS_FORM, DATASETS_FORM, INPUTS, OPTIMIZERS_LIST
from main import DATASETS_LIST, LEARNING_RATE_LIST, ACTIVATION_FUNC_LIST
from main import outputPanelSize, outputPanelRange, outputCanvasDensity
import output_panel
import model
import app


class MainFrame(wx.Frame):

    val_loss_text = "Test Loss: {:.3f}"
    loss_text = "Training Loss: {:.3f}"
    test_ratio_text = "Test Datasets Ratio: {:.2f}"
    noise_text = "Noise Level: {:d}"
    epoch_text = "Epoch: {:d}\nEpochs per sec: {:d}"
    bias_tooltip_text = "Bias is "

    border = 5
    mousePressed = False
    always_show_tooltip = False
    biasPanelMouseHoveredPos = None
    biasPanelMouseClickedPos = None
    neruonPanelMouseHoveredPos = None

    plot_counter = 0

    # Static property
    display_datas = copy.deepcopy(DATASETS_FORM)

    def __init__(self, main_app):
        super(MainFrame, self).__init__(None)
        self.app = main_app

        self.SetClientSize((DEFAULT_WIDTH, DEFAULT_HEIGHT))
        self.Center()
        self.SetTitle('Python Neural Network Playground')
        self.SetBackgroundColour('white')

        self.sizerFlags1 = wx.SizerFlags()
        self.sizerFlags1.Border(wx.ALL, self.border)
        self.sizerFlags2 = wx.SizerFlags()
        self.sizerFlags2.Expand().Border(wx.ALL, self.border)

        # Top
        topSizer = wx.BoxSizer(wx.HORIZONTAL)

        reset_button = wx.Button(self, label="Reset")
        reset_button.Bind(wx.EVT_BUTTON, self.resetButtonEvent)
        self.action_button = wx.Button(self, label="Start")
        self.action_button.Bind(wx.EVT_BUTTON, self.actionButtonEvent)
        self.epoch_statictext = wx.StaticText(
            self, label=self.epoch_text.format(0, 0))

        optimizerSizer = wx.BoxSizer(wx.VERTICAL)

        optimizer_list = [k for k in OPTIMIZERS_LIST]
        optimizer_statictext = wx.StaticText(self, label="Optimizer")
        optimizer_dropdown = wx.Choice(
            self, choices=optimizer_list, size=(100, -1))
        optimizer_dropdown.SetSelection(
            optimizer_list.index(self.app.model.choosed_optimizer))
        optimizer_dropdown.Bind(wx.EVT_CHOICE, self.optimizerDropdownEvent)

        optimizerSizer.Add(optimizer_statictext, 0, wx.TOP, self.border)
        optimizerSizer.Add(
            optimizer_dropdown, 0, wx.BOTTOM | wx.RIGHT, self.border)

        lrSizer = wx.BoxSizer(wx.VERTICAL)

        lr_str_list = list(map(str, LEARNING_RATE_LIST))

        lr_statictext = wx.StaticText(self, label="Learning Rate")
        lr_dropdown = wx.Choice(
            self, choices=lr_str_list, size=(100, -1))
        lr_dropdown.SetSelection(
            LEARNING_RATE_LIST.index(model.MainModel.learning_rate))
        lr_dropdown.Bind(wx.EVT_CHOICE, self.lrDropdownEvent)

        lrSizer.Add(lr_statictext, 0, wx.LEFT | wx.TOP, self.border)
        lrSizer.Add(
            lr_dropdown, 0, wx.LEFT | wx.BOTTOM | wx.RIGHT, self.border)

        topSizer.Add(reset_button, self.sizerFlags1)
        topSizer.Add(self.action_button, self.sizerFlags1)
        topSizer.Add(self.epoch_statictext, 1, wx.ALL, self.border)
        topSizer.Add(optimizerSizer)
        topSizer.Add(lrSizer)

        # Bottom
        bottomSizer = wx.FlexGridSizer(3)

        bottomSizer.AddMany([
            (self.registerBottomLeft()),
            (self.registerBottomCenter(), 0, wx.EXPAND),
            (self.registerBottomRight())])
        bottomSizer.AddGrowableCol(1)
        bottomSizer.AddGrowableRow(0)

        frameSizer = wx.BoxSizer(wx.VERTICAL)
        frameSizer.Add(topSizer, 1, wx.EXPAND)
        frameSizer.Add(bottomSizer, 8, wx.EXPAND)
        self.SetSizer(frameSizer)

        self.Bind(wx.EVT_MOTION, self.frameMotionEvent)
        self.Bind(wx.EVT_LEFT_UP, self.screenLeftReleasedEvent)

        self.frame_tooltip = wx.Panel(
            self, size=(150, 70), style=wx.BORDER_RAISED)
        self.frame_tooltip.SetBackgroundColour("white")
        self.frame_tooltip.Hide()

        tooltipSizer = wx.BoxSizer(wx.VERTICAL)

        frame_tooltip_statictext = wx.StaticText(
            self.frame_tooltip,
            label=self.bias_tooltip_text)
        frame_tooltip_statictext.SetForegroundColour("black")
        self.frame_tooltip_textctrl = wx.TextCtrl(
            self.frame_tooltip, size=(150, -1))
        self.frame_tooltip_textctrl.Bind(
            wx.EVT_TEXT, self.tooltipTextctrlTextEvent)

        tooltipSizer.Add(frame_tooltip_statictext, 0, wx.ALL, self.border)
        tooltipSizer.Add(
            self.frame_tooltip_textctrl, 0, wx.ALL, self.border)

        self.frame_tooltip.SetSizer(tooltipSizer)

    def registerBottomCenter(self):
        padding = 50

        scrolled_panel = wx.lib.scrolledpanel.ScrolledPanel(self)
        scrolled_panel.SetupScrolling()
        scrolled_panel.Bind(wx.EVT_MOTION, self.scrolledPanelMotionEvent)
        scrolled_panel.Bind(wx.EVT_LEFT_UP, self.screenLeftReleasedEvent)

        centerBottomSizer = wx.BoxSizer(wx.HORIZONTAL)

        inputsSizer = wx.BoxSizer(wx.VERTICAL)
        for input in INPUTS:
            inputs_checkbox = wx.CheckBox(scrolled_panel, label=input)
            if(input in model.MainModel.model_stru["inputs"]):
                inputs_checkbox.SetValue(True)
            inputs_checkbox.Bind(wx.EVT_CHECKBOX, self.inputsCheckboxEvent)
            inputsSizer.Add(inputs_checkbox, 0, wx.BOTTOM, padding)

        centerBottomSizer.Add(
            inputsSizer, 0, wx.RIGHT | wx.BOTTOM | wx.TOP, padding)

        self.act_func_list = [k for k in ACTIVATION_FUNC_LIST]
        i = 0
        self.neruon_panels = []
        self.bias_panels = []
        for hidden in model.MainModel.model_stru["hiddens"]:
            units = hidden["units"]
            hiddensSizer = wx.BoxSizer(wx.VERTICAL)

            units_num_statictext = wx.StaticText(
                scrolled_panel, label="({} neurons)".format(units))
            act_func_statictext = wx.StaticText(
                scrolled_panel, label="Activation Function")
            act_func_dropdown = wx.Choice(
                scrolled_panel, choices=self.act_func_list,
                size=(150, -1), name=str(i))
            act_func_dropdown.SetSelection(
                self.act_func_list.index(hidden["activation"]))
            act_func_dropdown.Bind(wx.EVT_CHOICE, self.actFuncDropdownEvent)
            hiddensSizer.Add(units_num_statictext, 0, wx.CENTER)
            hiddensSizer.Add(act_func_statictext)
            hiddensSizer.Add(act_func_dropdown, 0, wx.BOTTOM, 5)

            for unit in range(units):
                neruonSizer = wx.BoxSizer(wx.HORIZONTAL)

                bias_panel_name = str(i) + "," + str(unit)
                bias_panel = wx.Panel(
                    scrolled_panel, size=(5, 5), name=bias_panel_name)
                self.bias_panels.append(bias_panel)

                neruon_panel = output_panel.OutputPanel(
                    scrolled_panel, self,
                    50, self.app.output_neuron_density, name=bias_panel_name)
                neruon_panel.Bind(
                    wx.EVT_ENTER_WINDOW, self.neruonPanelMouseEnterEvent)
                neruon_panel.Bind(
                    wx.EVT_LEAVE_WINDOW, self.neruonPanelMouseLeaveEvent)
                self.neruon_panels.append(neruon_panel)

                neruonSizer.Add(bias_panel, 0, wx.ALIGN_BOTTOM | wx.RIGHT, 2)
                neruonSizer.Add(neruon_panel)

                hiddensSizer.Add(
                    neruonSizer, 0, wx.BOTTOM | wx.CENTER, padding)

            centerBottomSizer.Add(
                hiddensSizer, 0, wx.RIGHT, padding)
            i += 1

        scrolled_panel.SetSizer(centerBottomSizer)

        self.scrolled_panel_tooltip = wx.Panel(
            scrolled_panel, size=(150, 70), style=wx.BORDER_RAISED)
        self.scrolled_panel_tooltip.SetBackgroundColour("white")
        self.scrolled_panel_tooltip.Hide()

        tooltipSizer = wx.BoxSizer(wx.VERTICAL)

        scrolled_panel_tooltip_statictext = wx.StaticText(
            self.scrolled_panel_tooltip,
            label=self.bias_tooltip_text)
        scrolled_panel_tooltip_statictext.SetForegroundColour("black")
        self.scrolled_panel_tooltip_textctrl = wx.TextCtrl(
            self.scrolled_panel_tooltip, size=(150, -1))
        self.scrolled_panel_tooltip_textctrl.Bind(
            wx.EVT_TEXT, self.tooltipTextctrlTextEvent)

        tooltipSizer.Add(
            scrolled_panel_tooltip_statictext, 0, wx.ALL, self.border)
        tooltipSizer.Add(
            self.scrolled_panel_tooltip_textctrl, 0, wx.ALL, self.border)

        self.scrolled_panel_tooltip.SetSizer(tooltipSizer)
        return scrolled_panel

    def registerBottomLeft(self):
        leftBottomSizer = wx.BoxSizer(wx.VERTICAL)

        for each_dataset in DATASETS_LIST:
            dataset_button = wx.Button(self, label=each_dataset)
            dataset_button.Bind(wx.EVT_BUTTON, self.datasetButtonEvent)
            leftBottomSizer.Add(dataset_button, self.sizerFlags2)

        statictext_slider_font = wx.Font(
            8, wx.FONTFAMILY_DEFAULT,
            wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)

        self.test_ratio_slider_statictext = wx.StaticText(
            self,
            label=self.test_ratio_text.format(
                model.MainModel.test_datasets_ratio))
        self.test_ratio_slider_statictext.SetFont(statictext_slider_font)
        self.test_ratio_slider = wx.Slider(
            self, value=model.MainModel.test_datasets_ratio * 10,
            minValue=1, maxValue=9)
        self.test_ratio_slider.Bind(wx.EVT_SCROLL, self.testRatioSliderEvent)

        self.noise_slider_statictext = wx.StaticText(
            self,
            label=self.noise_text.format(app.MainApp.noise))
        self.noise_slider_statictext.SetFont(statictext_slider_font)
        self.noise_slider = wx.Slider(
            self, value=app.MainApp.noise,
            minValue=0, maxValue=10)
        self.noise_slider.Bind(wx.EVT_SCROLL, self.noiseSliderEvent)

        leftBottomSizer.Add(
            self.test_ratio_slider_statictext, self.sizerFlags1)
        leftBottomSizer.Add(self.test_ratio_slider, self.sizerFlags2)
        leftBottomSizer.Add(self.noise_slider_statictext, self.sizerFlags1)
        leftBottomSizer.Add(self.noise_slider, self.sizerFlags2)

        return leftBottomSizer

    def registerBottomRight(self):
        rightBottomSizer = wx.BoxSizer(wx.VERTICAL)

        lossesSizer = wx.BoxSizer(wx.HORIZONTAL)

        lossesTextSizer = wx.BoxSizer(wx.VERTICAL)

        statictext_loss_font = wx.Font(
            9, wx.FONTFAMILY_DEFAULT,
            wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)

        self.val_loss_statictext = wx.StaticText(
            self, label=self.val_loss_text.format(0))
        self.loss_statictext = wx.StaticText(
            self, label=self.loss_text.format(0))
        self.val_loss_statictext.SetFont(statictext_loss_font)
        self.loss_statictext.SetFont(statictext_loss_font)

        lossesTextSizer.Add(self.val_loss_statictext, self.sizerFlags1)
        lossesTextSizer.Add(self.loss_statictext, self.sizerFlags1)

        fig = figure(figsize=(1.7, 1))
        self.losses_axes = fig.add_axes([0, 0, 1, 1])
        self.losses_axes.set_axis_off()
        self.lineplot1, = self.losses_axes.plot([], [])
        self.lineplot2, = self.losses_axes.plot([], [])
        self.figure_canvas = FigureCanvas(self, wx.ID_ANY, fig)

        lossesSizer.Add(lossesTextSizer)
        lossesSizer.Add(self.figure_canvas, self.sizerFlags1)

        output_act_func_statictext = wx.StaticText(
            self, label="Activation Function")
        output_act_func_dropdown = wx.Choice(
            self, choices=self.act_func_list, name="output")

        output_act_func_dropdown.SetSelection(self.act_func_list.index(
            model.MainModel.model_stru["outputs"]["activation"]))
        output_act_func_dropdown.Bind(wx.EVT_CHOICE, self.actFuncDropdownEvent)

        self.output_panel = output_panel.OutputPanel(
            self, self, outputPanelSize, outputCanvasDensity)
        self.output_display_panel = wx.Panel(
            self.output_panel, size=self.output_panel.output_panel_size)
        self.output_display_panel.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.output_display_panel.Bind(wx.EVT_PAINT, self.onODPaint)

        self.output_display_panel.Bind(
            wx.EVT_LEFT_DOWN, self.displayPanelLeftPressedEvent)
        self.output_display_panel.Bind(
            wx.EVT_LEFT_UP, self.displayPanelLeftReleasedEvent)

        self.output_display_panel.Bind(
            wx.EVT_RIGHT_DOWN, self.displayPanelRightPressedEvent)
        self.output_display_panel.Bind(
            wx.EVT_RIGHT_UP, self.displayPanelRightReleasedEvent)

        self.output_display_panel.Bind(
            wx.EVT_MOTION, self.displayPanelMotionEvent)

        outputBiasPanelSizer = wx.BoxSizer(wx.HORIZONTAL)

        output_bias_panel_name = "output," + str(
            len(model.MainModel.model_stru["hiddens"])) + ",{}"
        self.bias_panels.append(
            wx.Panel(self, size=(5, 5), name=output_bias_panel_name.format(0)))
        outputBiasPanelSizer.Add(
            self.bias_panels[-1], 0, wx.LEFT | wx.BOTTOM, self.border)

        # (output range: [0 -> 1, 0 -> 1])
        if(MODE == 2):
            self.bias_panels.append(wx.Panel(
                self, size=(5, 5), name=output_bias_panel_name.format(1)))
            outputBiasPanelSizer.Add(
                self.bias_panels[-1], 0, wx.LEFT | wx.BOTTOM, self.border)

        for bias_panel in self.bias_panels:
            bias_panel.Bind(
                wx.EVT_ENTER_WINDOW, self.biasPanelMouseEnterEvent)
            bias_panel.Bind(
                wx.EVT_LEAVE_WINDOW, self.biasPanelMouseLeaveEvent)
            bias_panel.Bind(
                wx.EVT_LEFT_UP, self.biasPanelLeftReleasedEvent)

        self.color_explanation_panel = wx.Panel(
            self, size=(outputPanelSize, 10))
        self.color_explanation_panel.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.color_explanation_panel.Bind(wx.EVT_PAINT, self.onCEPaint)

        color_explanation_text_panel = wx.Panel(
            self, size=(outputPanelSize, 15))
        colorExplanationTextSizer = wx.BoxSizer(wx.HORIZONTAL)
        colorExplanationTextSizer.Add(
            wx.StaticText(color_explanation_text_panel, label="-1"), 1)
        colorExplanationTextSizer.Add(
            wx.StaticText(color_explanation_text_panel, label="0"), 1)
        colorExplanationTextSizer.Add(
            wx.StaticText(color_explanation_text_panel, label="1"), 0)
        color_explanation_text_panel.SetSizer(colorExplanationTextSizer)

        self.show_test_checkbox = wx.CheckBox(self, label="Show Test Data")
        self.show_test_checkbox.Bind(
            wx.EVT_CHECKBOX, self.showTestCheckboxEvent)
        self.discretize_checkbox = wx.CheckBox(self, label="Discretize Output")
        self.discretize_checkbox.Bind(
            wx.EVT_CHECKBOX, self.discretizeCheckboxEvent)
        self.custom_data_checkbox = wx.CheckBox(self, label="Custom Data")
        self.custom_data_checkbox.Bind(
            wx.EVT_CHECKBOX, self.customDataCheckboxEvent)

        rightBottomSizer.Add(lossesSizer)
        rightBottomSizer.Add(
            output_act_func_statictext, 0,
            wx.LEFT | wx.TOP, self.border)
        rightBottomSizer.Add(
            output_act_func_dropdown, 0,
            wx.EXPAND | wx.RIGHT | wx.LEFT | wx.BOTTOM, self.border)
        rightBottomSizer.Add(self.output_panel, self.sizerFlags1)
        rightBottomSizer.Add(outputBiasPanelSizer)
        rightBottomSizer.Add(
            self.color_explanation_panel, 0,
            wx.LEFT | wx.TOP, self.border)
        rightBottomSizer.Add(
            color_explanation_text_panel, 0,
            wx.LEFT | wx.BOTTOM, self.border)

        sizerFlags3 = wx.SizerFlags()
        sizerFlags3.Border(wx.BOTTOM | wx.TOP, self.border)
        rightBottomSizer.Add(self.show_test_checkbox, sizerFlags3)
        rightBottomSizer.Add(self.discretize_checkbox, sizerFlags3)
        rightBottomSizer.Add(self.custom_data_checkbox, sizerFlags3)

        self.val_loss_statictext.SetForegroundColour(
            self.lineplot1.get_color())
        self.loss_statictext.SetForegroundColour(self.lineplot2.get_color())

        return rightBottomSizer

    def addCustomData(self, color, pos):
        mouse_x, mouse_y = pos
        x = numpy.interp(
            mouse_x, [0, outputPanelSize - 1],
            outputPanelRange)
        y = numpy.interp(
            mouse_y, [0, outputPanelSize - 1],
            [outputPanelRange[1], outputPanelRange[0]])
        MainFrame.display_datas["train"].append([mouse_x, mouse_y, color])

        y_train = color
        if(MODE == 2):
            y_train = [1, 0] if(color == 1) else [0, 1]  # (output range: [0 -> 1, 0 -> 1])
        model.MainModel.datasets["train"].append([[x, y], y_train])

        self.app.model.updateDatasets()
        self.output_display_panel.Refresh()

    def addLossesPlotData(self, x, val_loss, loss):
        lp1_ydata = numpy.append(self.lineplot1.get_ydata(), val_loss)
        lp2_ydata = numpy.append(self.lineplot2.get_ydata(), loss)

        self.lineplot1.set_data(
            numpy.append(self.lineplot1.get_xdata(), x), lp1_ydata)

        self.lineplot2.set_data(
            numpy.append(self.lineplot2.get_xdata(), x), lp2_ydata)

        self.losses_axes.set_xlim(-1, self.plot_counter)
        self.losses_axes.set_ylim(0, max(max(lp1_ydata), max(lp2_ydata)))
        self.figure_canvas.draw()

        self.plot_counter += 0.1

    def resetLossesPlotData(self):
        self.lineplot1.set_data([], [])
        self.lineplot2.set_data([], [])
        self.figure_canvas.draw()

        self.plot_counter = 0

    def updateTrainText(self, event=None):
        self.epoch_statictext.SetLabel(
            self.epoch_text.format(app.MainApp.epochs, app.MainApp.eps))
        logs = app.MainApp.logs
        if("val_loss" not in logs):
            logs["val_loss"] = 0

        self.val_loss_statictext.SetLabel(
            self.val_loss_text.format(logs["val_loss"]))
        self.loss_statictext.SetLabel(
            self.loss_text.format(logs["loss"]))

        self.addLossesPlotData(
            self.plot_counter, logs["val_loss"], logs["loss"])

        weights = self.app.model.get_weights()
        if(self.biasPanelMouseHoveredPos is not None):
            i, j = self.biasPanelMouseHoveredPos
            bias = weights[i][j]
            self.frame_tooltip_textctrl.ChangeValue(str(bias))
            self.scrolled_panel_tooltip_textctrl.ChangeValue(str(bias))
        index = 1
        i = 0
        for j in range(len(model.MainModel.model_stru["hiddens"]) + 1):
            bias_list = weights[index]
            for bias in bias_list:
                if(MODE == 0):
                    map_bias = max(-1, min(1, bias))  # (output range: -1 -> 1)
                if(MODE == 1 or MODE == 2):
                    map_bias = numpy.interp(bias, [-1, 1], [0, 1])  # (output range: 0 -> 1 and [0 -> 1, 0 -> 1])
                self.bias_panels[i].SetBackgroundColour(
                    self.app.mapOutputToRGBA(map_bias))
                i += 1
            index += 2

    # Button event
    def actionButtonEvent(self, event):
        model.MainModel.is_training = not model.MainModel.is_training
        if model.MainModel.is_training:
            self.action_button.SetLabel("Stop")
            self.app.setupAndStartThread()
        else:
            self.action_button.SetLabel("Start")

    def resetButtonEvent(self, event):
        if model.MainModel.is_training:
            self.actionButtonEvent(None)

        self.app.resetModel()

    def datasetButtonEvent(self, event):
        self.app.model.choosed_pattern = event.GetEventObject().GetLabel()
        if(not app.MainApp.custom_data):
            self.updateDatasetsAndOutputDisplay()

    # Checkbox event
    def showTestCheckboxEvent(self, event):
        app.MainApp.display_test = self.show_test_checkbox.IsChecked()
        self.output_display_panel.Refresh()

    def discretizeCheckboxEvent(self, event):
        app.MainApp.display_discretize = self.discretize_checkbox.IsChecked()
        if(not model.MainModel.is_training):
            self.app.model.predictToNeuronsCanvas(newpredict=False)
            self.app.model.predictToOutputCanvas(newpredict=False)

    def customDataCheckboxEvent(self, event):
        app.MainApp.custom_data = self.custom_data_checkbox.IsChecked()
        if(app.MainApp.custom_data):
            MainFrame.display_datas = copy.deepcopy(DATASETS_FORM)
            model.MainModel.datasets = copy.deepcopy(DATASETS_FORM)
            model.MainModel.datasets4train = copy.deepcopy(TRAIN_DATASETS_FORM)
            self.output_display_panel.Refresh()
        else:
            self.updateDatasetsAndOutputDisplay()

    def inputsCheckboxEvent(self, event):
        checkbox = event.GetEventObject()

        label = checkbox.GetLabel()
        if(checkbox.IsChecked()):
            if(label not in model.MainModel.model_stru["inputs"]):
                model.MainModel.model_stru["inputs"].append(label)
        else:
            if(label in model.MainModel.model_stru["inputs"]):
                model.MainModel.model_stru["inputs"].remove(label)

        self.doAfterModelStructureChanged()

    # Slider event
    def testRatioSliderEvent(self, event):
        model.MainModel.test_datasets_ratio = event.GetPosition() * 0.1
        self.test_ratio_slider_statictext.SetLabel(
            self.test_ratio_text.format(model.MainModel.test_datasets_ratio))
        if(not app.MainApp.custom_data):
            self.updateDatasetsAndOutputDisplay()

    def noiseSliderEvent(self, event):
        app.MainApp.noise = event.GetPosition()
        self.noise_slider_statictext.SetLabel(
            self.noise_text.format(app.MainApp.noise))
        if(not app.MainApp.custom_data):
            self.updateDatasetsAndOutputDisplay()

    # Choice event
    def lrDropdownEvent(self, event):
        model.MainModel.learning_rate = LEARNING_RATE_LIST[event.GetInt()]
        if not hasattr(self.app.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        backend.set_value(
            self.app.model.optimizer.lr, model.MainModel.learning_rate)

    def optimizerDropdownEvent(self, event):
        self.app.model.choosed_optimizer = event.GetString()
        self.doAfterModelStructureChanged()

    def actFuncDropdownEvent(self, event):
        hidden_index = event.GetEventObject().GetName()
        activation_name = event.GetString()
        if(hidden_index == "output"):
            model.MainModel.model_stru[
                "outputs"]["activation"] = activation_name
        else:
            hidden_index = int(hidden_index)
            model.MainModel.model_stru[
                "hiddens"][hidden_index]["activation"] = activation_name

        self.doAfterModelStructureChanged()

    # Mouse event
    def displayPanelLeftPressedEvent(self, event):
        event.Skip()
        if(app.MainApp.custom_data):
            self.mousePressed = True
            self.color = 1
            self.addCustomData(self.color, event.GetPosition())

    def displayPanelLeftReleasedEvent(self, event):
        if(app.MainApp.custom_data):
            self.mousePressed = False

    def displayPanelRightPressedEvent(self, event):
        event.Skip()
        if(app.MainApp.custom_data):
            self.mousePressed = True
            self.color = -1
            self.addCustomData(self.color, event.GetPosition())

    def displayPanelRightReleasedEvent(self, event):
        if(app.MainApp.custom_data):
            self.mousePressed = False

    def displayPanelMotionEvent(self, event):
        if(app.MainApp.custom_data and self.mousePressed):
            self.addCustomData(self.color, event.GetPosition())

    def scrolledPanelMotionEvent(self, event):
        if(not self.always_show_tooltip):
            self.scrolled_panel_tooltip.SetPosition(
                event.GetPosition() + (10, 10))

    def frameMotionEvent(self, event):
        if(not self.always_show_tooltip):
            self.frame_tooltip.SetPosition(event.GetPosition() + (10, 10))

    def biasPanelMouseEnterEvent(self, event):
        split_name = event.GetEventObject().GetName().split(',')
        if(split_name[0] == "output"):
            pos = [int(split_name[1]), int(split_name[2])]
            textctrl = self.frame_tooltip_textctrl
            tooltip = self.frame_tooltip
        else:
            pos = [int(split_name[0]), int(split_name[1])]
            textctrl = self.scrolled_panel_tooltip_textctrl
            tooltip = self.scrolled_panel_tooltip

        pos[0] = pos[0] * 2 + 1
        self.biasPanelMouseHoveredPos = pos

        weights = self.app.model.get_weights()
        textctrl.ChangeValue(str(weights[pos[0]][pos[1]]))
        tooltip.Show()

    def biasPanelMouseLeaveEvent(self, event):
        self.biasPanelMouseHoveredPos = None
        if(not self.always_show_tooltip):
            self.scrolled_panel_tooltip.Hide()
            self.frame_tooltip.Hide()

    def biasPanelLeftReleasedEvent(self, event):
        self.always_show_tooltip = True
        self.biasPanelMouseClickedPos = self.biasPanelMouseHoveredPos

    def screenLeftReleasedEvent(self, event):
        self.always_show_tooltip = False
        self.biasPanelMouseClickedPos = None
        self.biasPanelMouseLeaveEvent(None)

    def neruonPanelMouseEnterEvent(self, event):
        neruon_panel_name = event.GetEventObject().GetName()
        pos = list(map(int, neruon_panel_name.split(',')))
        self.neruonPanelMouseHoveredPos = pos
        if(not model.MainModel.is_training):
            self.app.model.predictToOutputCanvas()

    def neruonPanelMouseLeaveEvent(self, event):
        self.neruonPanelMouseHoveredPos = None
        if(not model.MainModel.is_training):
            self.app.model.predictToOutputCanvas()

    # Textctrl event
    def tooltipTextctrlTextEvent(self, event):
        weights = self.app.model.get_weights()
        i, j = self.biasPanelMouseClickedPos

        textctrl = event.GetEventObject()
        try:
            custom_bias = float(textctrl.GetValue())
            weights[i][j] = custom_bias
            self.app.model.set_weights(weights)

            if(not model.MainModel.is_training):
                self.app.model.predictToNeuronsCanvas()
                self.app.model.predictToOutputCanvas()
        except ValueError:
            textctrl.ChangeValue(str(weights[i][j]))

    # Paint event
    def onCEPaint(self, event):
        dc = wx.AutoBufferedPaintDC(self.color_explanation_panel)
        w, h = event.GetEventObject().GetClientSize()

        dc.GradientFillLinear(
            (0, 0, w * 0.5, h),
            self.app.output_color[-1], self.app.output_color[0])

        dc.GradientFillLinear(
            (w * 0.5, 0, w, h),
            self.app.output_color[0], self.app.output_color[1])

    def onODPaint(self, event):
        w, h = event.GetEventObject().GetClientSize()
        middle_x = w * 0.5
        middle_y = h * 0.5

        dc = wx.AutoBufferedPaintDC(self.output_display_panel)

        # Draw a coordinate line
        dc.SetPen(wx.Pen((207, 207, 196), 3, wx.PENSTYLE_DOT))
        dc.DrawLine(middle_x, 0, middle_x, w)
        dc.DrawLine(0, middle_y, h, middle_y)

        # Draw datasets point
        if(MainFrame.display_datas["train"] is not None):
            for data in MainFrame.display_datas["train"]:
                index = int(
                    app.MainApp.half_step + data[2] * app.MainApp.half_step)
                color = list(app.MainApp.output_color_step[index])
                color[3] = 255
                dc.SetPen(
                    wx.Pen(tuple(color), 5))
                dc.DrawPoint(data[0], data[1])

        if(
            app.MainApp.display_test
            and MainFrame.display_datas["test"] is not None
        ):
            for data in MainFrame.display_datas["test"]:
                index = int(
                    app.MainApp.half_step + data[2] * app.MainApp.half_step)
                color = list(app.MainApp.output_color_step[index])
                color[3] = 255
                dc.SetPen(
                    wx.Pen(tuple(color), 7))
                dc.DrawPoint(data[0], data[1])

    # Utils
    def updateDatasetsAndOutputDisplay(self, newrandom=True):
        if(newrandom):
            self.app.model.randomDatasets()
        self.app.model.updateDatasets()
        self.output_display_panel.Refresh()

    def doAfterModelStructureChanged(self):
        if model.MainModel.is_training:
            self.actionButtonEvent(None)

        self.updateDatasetsAndOutputDisplay(newrandom=False)
        self.app.model.preparePredictDataForCanvas()
        self.app.resetModel(recompile=True)
