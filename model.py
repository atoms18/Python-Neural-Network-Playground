
import random
import copy

import wx
import numpy
from keras import backend
from keras.models import Model
from keras.layers import Dense, Activation, Input

from main import DATASETS_NUM, DEFAULT_MODEL, INPUTS
from main import TRAIN_DATASETS_FORM, DATASETS_FORM
from main import OPTIMIZERS_LIST, DATASETS_LIST, ACTIVATION_FUNC_LIST
from main import outputPanelSize, outputPanelRange, outputCanvasDensity
import callback
import frame
import app


class MainModel(Model):
    session = backend.get_session()

    ml_callback = callback.Callback()

    # Static property
    model_stru = copy.deepcopy(DEFAULT_MODEL)

    learning_rate = 3e-2
    test_datasets_ratio = 0.2

    is_training = False

    def __init__(self, main_app):
        self.graph = self.session.graph
        self.app = main_app
        self.choosed_pattern = "Circle"
        self.choosed_optimizer = "Adam"

        canvas_pos4canvas = []
        for j in range(outputCanvasDensity):
            for i in range(outputCanvasDensity):
                x_for_ml = numpy.interp(
                    i, [0, outputCanvasDensity - 1], outputPanelRange)
                y_for_ml = numpy.interp(
                    j, [0, outputCanvasDensity - 1],
                    [outputPanelRange[1], outputPanelRange[0]])
                canvas_pos4canvas.append([x_for_ml, y_for_ml])
        self.canvas_pos4canvas = canvas_pos4canvas

        canvas_pos4neuron = []
        for j in range(self.app.output_neuron_density):
            for i in range(self.app.output_neuron_density):
                x_for_ml = numpy.interp(
                    i, [0, self.app.output_neuron_density - 1],
                    outputPanelRange)
                y_for_ml = numpy.interp(
                    j, [0, self.app.output_neuron_density - 1],
                    [outputPanelRange[1], outputPanelRange[0]])
                canvas_pos4neuron.append([x_for_ml, y_for_ml])
        self.canvas_pos4neuron = canvas_pos4neuron
        self.preparePredictDataForCanvas()

        self.randomDatasets()
        self.reCreateModel()

    def preparePredictDataForCanvas(self):
        self.predict_data4canvas = numpy.array(
            list(map(self.getInputsForm, self.canvas_pos4canvas)))
        self.predict_data4neuron = numpy.array(
            list(map(self.getInputsForm, self.canvas_pos4neuron)))

    def reCreateModel(self):
        inputs = Input(shape=(len(MainModel.model_stru["inputs"]),))

        h = inputs
        for hidden in MainModel.model_stru["hiddens"]:
            h = Dense(hidden["units"])(h)
            act_func = hidden["activation"]
            if(callable(act_func)):
                h = Activation(act_func)(h)
            elif(act_func in ACTIVATION_FUNC_LIST):
                h = Activation(ACTIVATION_FUNC_LIST[act_func])(h)

        outputs = Dense(1)(h)
        # outputs = Dense(2)(h)  # (output range: [0 -> 1, 0 -> 1])

        act_func = MainModel.model_stru["outputs"]["activation"]
        if(callable(act_func)):
            outputs = Activation(act_func)(outputs)
        elif(act_func in ACTIVATION_FUNC_LIST):
            outputs = Activation(ACTIVATION_FUNC_LIST[act_func])(outputs)

        super(MainModel, self).__init__(inputs=inputs, outputs=outputs)

        self.compile(
            optimizer=OPTIMIZERS_LIST[self.choosed_optimizer](
                lr=MainModel.learning_rate),
            loss='mean_squared_error',
            metrics=['accuracy'])

        index = 2
        self.hiddens_layer_model = []
        for i in range(len(MainModel.model_stru["hiddens"])):
            hidden_layer_model = Model(
                inputs=self.input, outputs=self.layers[index].output)
            self.hiddens_layer_model.append(hidden_layer_model)
            index += 2

        self.initLoss()

    def initLoss(self):
        val_loss = 0
        loss = 0
        check = True
        if(app.MainApp.custom_data):
            train_num = len(MainModel.datasets4train["X_train"])
            if(train_num <= 0):
                check = False
        else:
            val_loss, val_acc = self.evaluate(
                MainModel.datasets4train["X_test"],
                MainModel.datasets4train["Y_test"], verbose=0)

        if(check):
            loss, acc = self.evaluate(
                MainModel.datasets4train["X_train"],
                MainModel.datasets4train["Y_train"], verbose=0)
        app.MainApp.logs["val_loss"] = val_loss
        app.MainApp.logs["loss"] = loss

    def randomDatasets(self):
        frame.MainFrame.display_datas = copy.deepcopy(DATASETS_FORM)
        datasets_ = copy.deepcopy(DATASETS_FORM)

        test_num = int(DATASETS_NUM * MainModel.test_datasets_ratio)
        args = [
            random.randint(0, 2), 0, app.MainApp.noise * 0.1, 0, "", 0]

        train_i = 0
        test_i = 0
        for i in range(DATASETS_NUM):
            if(i > test_num):
                dataset_suffix = "train"
                max_num = abs(DATASETS_NUM - test_num)
                current_i = train_i
                train_i += 1
            else:
                dataset_suffix = "test"
                max_num = test_num
                current_i = test_i
                test_i += 1

            args[3] = current_i
            args[4] = dataset_suffix
            args[5] = max_num
            isPass = False
            while(not isPass):
                x_rand = random.uniform(
                    outputPanelRange[0], outputPanelRange[1])
                y_rand = random.uniform(
                    outputPanelRange[0], outputPanelRange[1])

                apply_to_pattern = DATASETS_LIST[self.choosed_pattern](
                    x_rand, y_rand, args)
                if(apply_to_pattern is not False):
                    new_x, new_y, color = apply_to_pattern
                    if(
                        new_x > outputPanelRange[1]
                        or new_x < outputPanelRange[0]
                    ):
                        continue
                    if(
                        new_y > outputPanelRange[1]
                        or new_y < outputPanelRange[0]
                    ):
                        continue

                    if(isinstance(color, float)):
                        y_train = color
                        # if(color > 0):
                        #     y_train = [color, 0]
                        # else:
                        #     y_train = [0, -color]
                    else:
                        y_train = color
                        # y_train = [1, 0] if(color == 1) else [0, 1]  # (output range: [0 -> 1, 0 -> 1])

                    datasets_[dataset_suffix].append([[new_x, new_y], y_train])

                    x_map = numpy.interp(
                        new_x, outputPanelRange, [0, outputPanelSize - 1])
                    y_map = numpy.interp(
                        new_y, outputPanelRange, [outputPanelSize - 1, 0])
                    frame.MainFrame.display_datas[dataset_suffix].append(
                        [x_map, y_map, color])
                    isPass = True

        MainModel.datasets = datasets_
        self.updateDatasets()

    def updateDatasets(self):
        datasets_ = copy.deepcopy(TRAIN_DATASETS_FORM)
        for dataset_key in MainModel.datasets:
            datasets = MainModel.datasets[dataset_key]
            for each_dataset in datasets:
                x = each_dataset[0][0]
                y = each_dataset[0][1]
                color = each_dataset[1]

                datasets_["X_" + dataset_key].append(
                    self.getInputsForm([x, y]))

                # y_train = [color]  # (output range: -1 -> 1)
                y_train = numpy.interp(color, [-1, 1], [0, 1])  # (output range: 0 -> 1)
                # y_train = color  # (output range: [0 -> 1, 0 -> 1])
                datasets_["Y_" + dataset_key].append(y_train)

        datasets_["X_train"] = numpy.array(datasets_["X_train"])
        datasets_["Y_train"] = numpy.array(datasets_["Y_train"])
        datasets_["X_test"] = numpy.array(datasets_["X_test"])
        datasets_["Y_test"] = numpy.array(datasets_["Y_test"])
        MainModel.datasets4train = datasets_

    def predictToOutputCanvas(self, newpredict=True):
        neruon_pos = self.app.main_frame.neruonPanelMouseHoveredPos
        if(neruon_pos is not None):
            i, j = neruon_pos
            predict_neroun = self.hiddens_layer_model[i].predict_on_batch(
                self.predict_data4canvas).T
            self.current_predict_datas = predict_neroun[j]
        else:
            if(newpredict):
                self.current_predict_datas = self.predict_on_batch(
                    self.predict_data4canvas)

        map_predict_datas_to_rgba = list(map(
            self.app.mapOutputToRGBA, self.current_predict_datas))
        wx.CallAfter(
            self.app.main_frame.output_panel.updateOutputCanvasImage,
            map_predict_datas_to_rgba)
        return self.current_predict_datas

    def predictToNeuronsCanvas(self, newpredict=True):
        if(newpredict):
            datas = []
            for model in self.hiddens_layer_model:
                predicts = model.predict_on_batch(self.predict_data4neuron).T
                for predict in predicts:
                    datas.append(predict)
            self.current_neuron_predict_datas = datas

        map_predict_datas_to_rgba = list(map(
            self.app.mapOutputListToRGBAList,
            self.current_neuron_predict_datas))

        panels = self.app.main_frame.neruon_panels
        for panel_index in range(len(panels)):
            wx.CallAfter(
                panels[panel_index].updateOutputCanvasImage,
                map_predict_datas_to_rgba[panel_index])
        return self.current_neuron_predict_datas

    # Utils
    def getInputsForm(self, pos):
        inputs = []
        for input in MainModel.model_stru["inputs"]:
            inputs.append(INPUTS[input](pos[0], pos[1]))
        return inputs

    def stopTraining(self):
        MainModel.is_training = False
