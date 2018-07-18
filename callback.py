
import time

from keras.callbacks import Callback

import model
import app


class Callback(Callback):
    eps_counter = 0
    eps_time_counter = time.time()

    def on_epoch_end(self, epoch, logs={}):
        if(model.MainModel.is_training):
            app.MainApp.logs = logs

            diff = time.time() - self.eps_time_counter
            if(diff > 1):
                app.MainApp.eps = int(self.eps_counter / diff)
                self.eps_counter = 0
                self.eps_time_counter = time.time()

            self.eps_counter += 1

            app.MainApp.epochs += 1

    def on_batch_end(self, batch, logs={}):
        if(not model.MainModel.is_training):
            self.model.stop_training = True
