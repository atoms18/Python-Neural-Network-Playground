
import threading
import sys
import os

import wx
import numpy

import frame
import model
import thread


class MainApp(wx.App):

    output_color = {
        -1: (197, 109, 109),
        1: (8, 119, 189),
        0: (232, 234, 235)}
    # output_color = {
    #     -1: (255, 0, 0),
    #     1: (0, 0, 255),
    #     0: (255, 255, 255)}
    output_neuron_density = 25

    # Static property
    logs = {}

    color_step = 30
    noise = 0
    eps = 0
    epochs = 0

    custom_data = False
    display_test = False
    display_discretize = False

    def __init__(self):
        super(MainApp, self).__init__(False)

        self.model = model.MainModel(self)

        def subtractList(l1, l2):
            return [a - b for (a, b) in zip(l1, l2)]

        alpha = 160
        MainApp.half_step = MainApp.color_step * 0.5
        output_color_step = []
        middle_color = self.output_color[0]
        for i in numpy.arange(-1, 1 + 1e-9, 1 / MainApp.half_step):
            index = -1 if(i < 0) else 1
            each_color = self.output_color[index]
            if(index is -1):
                each_color = subtractList(middle_color, each_color)
            else:
                each_color = subtractList(each_color, middle_color)

            new_color = map(int, [
                middle_color[0] + each_color[0] * i,
                middle_color[1] + each_color[1] * i,
                middle_color[2] + each_color[2] * i, alpha])

            output_color_step.append(tuple(new_color))
        MainApp.output_color_step = output_color_step

        self.main_frame = frame.MainFrame(self)
        self.model.predictToNeuronsCanvas()
        self.model.predictToOutputCanvas()
        self.main_frame.updateTrainText()
        self.main_frame.Show()

    def setupAndStartThread(self):
        train_thread = threading.Thread(
            target=thread.train_thread,
            args=[self])
        train_thread.start()
        predict_thread = threading.Thread(
            target=thread.predict_thread,
            args=[self])
        predict_thread.start()

    # Utils
    def resetModel(self, recompile=False):
        MainApp.epochs = 0
        MainApp.eps = 0
        if(recompile):
            self.model.reCreateModel()
        else:
            for layer in self.model.layers:
                if hasattr(layer, 'kernel_initializer'):
                    layer.kernel.initializer.run(session=self.model.session)
                if hasattr(layer, 'bias_initializer'):
                    layer.bias.initializer.run(session=self.model.session)
            self.model.initLoss()
        self.main_frame.updateTrainText()
        self.main_frame.resetLossesPlotData()
        self.model.predictToNeuronsCanvas()
        self.model.predictToOutputCanvas()

    def mapOutputToRGBA(self, output):

        # (output range: -1 -> 1)
        # index = int(MainApp.half_step + output * MainApp.half_step)
        # if(MainApp.display_discretize):
        #     index = 0 if output < 0 else MainApp.color_step

        # (output range: 0 -> 1)
        index = max(
            0, min(MainApp.color_step, int(output * MainApp.color_step)))
        if(MainApp.display_discretize):
            index = 0 if output < 0.5 else MainApp.color_step

        # (output range: [0 -> 1, 0 -> 1])
        # diff = abs(output[0]) - abs(output[1])
        #
        # index = int(MainApp.half_step + diff * MainApp.half_step)
        # if(MainApp.display_discretize):
        #     index = 0 if(diff < 0) else MainApp.color_step

        color = MainApp.output_color_step[index]
        return color

    def mapOutputListToRGBAList(self, output):
        return list(map(self.mapOutputToRGBA, output))

    @staticmethod
    def printError(error=None):
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        if(error is not None):
            print(error)
