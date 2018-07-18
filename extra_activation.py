
import tensorflow


class ExtraActivation():
    def gaussian(x):
        return tensorflow.exp(-tensorflow.pow(x, 2))

    def sin(x):
        return tensorflow.sin(x)
