import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.models.rnn.rnn_cell import linear


class Projector(object):
    def __init__(self, to_size, bias=False, non_linearity=None):
        self.to_size = to_size
        self.bias = bias
        self.non_linearity = non_linearity

    def __call__(self, inputs, scope=None):
        """
        :param inputs: list of 2D Tensors with shape [batch_size x self.from_size]
        :return: list of 2D Tensors with shape [batch_size x self.to_size]
        """
        with vs.variable_scope(scope or "Projector"):
            projected = linear(inputs, self.to_size, self.bias)
            if self.non_linearity is not None:
                projected = self.non_linearity(projected)
        return projected