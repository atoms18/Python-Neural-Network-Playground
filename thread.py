
import time

import wx

from main import DEBUG
import model
import app


def train_thread(app_):
    with app_.model.graph.as_default():
        time.sleep(0.1)
        while model.MainModel.is_training:
            try:
                # loss, acc = app_.model.train_on_batch(
                #     model.MainModel.datasets4train["X_train"],
                #     model.MainModel.datasets4train["Y_train"])
                # if(not app.MainApp.custom_data):
                #     val_loss, val_acc = app_.model.evaluate(
                #         model.MainModel.datasets4train["X_test"],
                #         model.MainModel.datasets4train["Y_test"], verbose=0)
                #     app.MainApp.logs["val_loss"] = val_loss
                #     app.MainApp.logs["loss"] = loss
                #     # print("val_loss = {:.4f}, val_acc = {:.4f}".format(
                #     #     val_loss, val_acc))
                # # else:
                #     # print("loss = {:.4f}, acc = {:.4f}".format(loss, acc))
                # app_.model.ml_callback.on_epoch_end(0)

                if(not app.MainApp.custom_data):
                    app_.model.fit(
                       model.MainModel.datasets4train["X_train"],
                       model.MainModel.datasets4train["Y_train"],
                       batch_size=64, epochs=10, verbose=0,
                       callbacks=[app_.model.ml_callback],
                       validation_data=(
                           model.MainModel.datasets4train["X_test"],
                           model.MainModel.datasets4train["Y_test"]))
                else:
                    train_num = len(model.MainModel.datasets4train["X_train"])
                    print("Train with: {}".format(train_num))
                    if(train_num > 0):
                        app_.model.fit(
                           model.MainModel.datasets4train["X_train"],
                           model.MainModel.datasets4train["Y_train"],
                           batch_size=64, epochs=10, verbose=0,
                           callbacks=[app_.model.ml_callback])

                # if(app.MainApp.epochs is 100):
                #     app_.model.save("100_epochs_model.h5")
            except Exception as e:
                if(DEBUG):
                    print("Train thread error")
                    app_.printError(e)
                continue


def predict_thread(app_):
    with app_.model.graph.as_default():
        time.sleep(0.5)

        fps_counter = 0
        fps_start_time = time.time()
        while model.MainModel.is_training:
            try:
                app_.model.predictToNeuronsCanvas()
                app_.model.predictToOutputCanvas()
                wx.CallAfter(app_.main_frame.updateTrainText)

                diff = time.time() - fps_start_time
                if(diff > 1):
                    print("Render FPS: {}".format(fps_counter / diff))
                    fps_counter = 0
                    fps_start_time = time.time()

                fps_counter += 1

                time.sleep(1 / 60)
            except Exception as e:
                if(DEBUG):
                    print("Predicts thread error")
                    app_.printError(e)
                continue
