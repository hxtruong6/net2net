import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras import Model
import json

from net2net import net2deeper, net2wider


class Layers(Layer):
    def __init__(self, hidden_dims=None):
        super(Layers, self).__init__()
        self.layers = []
        for i, dim in enumerate(hidden_dims):
            layer = Dense(
                units=dim,
                activation=tf.nn.relu,
            )
            self.layers.append(layer)

    def wider(self, added_size=1, pos_layer=None):
        layers_size = len(self.layers)
        if layers_size < 2:
            raise ValueError("Number of layer must be greater than 2.")
        if pos_layer is None:
            pos_layer = max(layers_size - 2, 0)
        elif pos_layer >= layers_size - 1 or pos_layer < 0:
            raise ValueError(
                f"pos_layer is expected less than length of layers (pos_layer in [0, layers_size-2])")

        weights, bias = self.layers[pos_layer].get_weights()
        weights_next_layer, bias_next_layer = self.layers[pos_layer + 1].get_weights()

        new_weights, new_bias, new_weights_next_layer = net2wider(weights, bias, weights_next_layer, added_size)

        src_units, des_units = weights.shape[0], weights.shape[1] + added_size
        next_des_units = weights_next_layer.shape[1]

        wider_layer = Dense(
            units=des_units,
            activation=tf.nn.relu,
        )

        # input_shape = (batch_size, input_features).
        # input_features = number of units in layer = length(layer) = output of previous layer
        wider_layer.build(input_shape=(1, src_units))
        wider_layer.set_weights([new_weights, new_bias])

        next_layer = Dense(
            units=next_des_units,
            activation=tf.nn.relu,
        )
        next_layer.build(input_shape=(1, des_units))
        next_layer.set_weights([new_weights_next_layer, bias_next_layer])

        self.layers[pos_layer] = wider_layer
        self.layers[pos_layer + 1] = next_layer

    def deeper(self, pos_layer=None):
        layers_size = len(self.layers)
        if pos_layer is None:
            pos_layer = max(layers_size - 1, 0)
        elif pos_layer >= layers_size - 1 or pos_layer < 0:
            raise ValueError(
                f"pos_layer is expected less than length of layers (pos_layer in [0, layers_size-2]).")

        weights, bias = self.layers[pos_layer].get_weights()
        new_weights, new_bias = net2deeper(weights)
        des_units = weights.shape[1]
        layer = Dense(
            units=des_units,
            activation=tf.nn.relu,
        )
        layer.build(input_shape=(1, des_units))
        layer.set_weights([new_weights, new_bias])

        self.layers.insert(pos_layer + 1, layer)

    def set_dump_weight(self):
        for i in range(len(self.layers)):
            w, b = self.layers[i].get_weights()

            for u in range(w.shape[0]):
                for v in range(w.shape[1]):
                    w[u][v] = u * w.shape[1] + v
            for v in range(b.shape[0]):
                b[v] = v

            self.layers[i].set_weights([w, b])

    def call(self, inputs):
        z = inputs
        for layer in self.layers:
            z = layer(z)

        return z

    def info(self, show_weight=False, show_config=False):
        print(f"{self.name}\n----------")
        print(f"Number of layers: {len(self.layers)}")
        for i, layer in enumerate(self.layers):
            print(f"Layer {i + 1}\n\t Name={layer.name}\n\t Shape ={layer.get_weights()[0].shape}")
            if show_weight:
                print(f"\t Weight= {layer.get_weights()}")
            if show_config:
                print(f"Config: {json.dumps(layer.get_config(), sort_keys=True, indent=4)}")

    def get_length_layers(self):
        return len(self.layers)


class MyModel(Model):
    def __init__(self, hidden_dims=None, v1=0.01, v2=0.01):
        super(MyModel, self).__init__()

        if hidden_dims is None:
            hidden_dims = [256, 512]

        self.custom_layers = Layers(hidden_dims=hidden_dims)

    def wider(self, added_size=1, pos_layer=None):
        if pos_layer is None:
            pos_layer = self.custom_layers.get_length_layers() - 2

        self.custom_layers.wider(added_size=added_size, pos_layer=pos_layer)

    def deeper(self, pos_layer=None):
        if pos_layer is None:
            pos_layer = self.custom_layers.get_length_layers() - 2

        self.custom_layers.deeper(pos_layer=pos_layer)

    def call(self, inputs):
        return self.custom_layers(inputs)

    def get_output(self, inputs):
        return self.custom_layers(inputs)

    def info(self, show_weight=False, show_config=False):
        self.custom_layers.info(show_weight, show_config)


if __name__ == "__main__":
    my_model = MyModel(hidden_dims=[3, 2])
    X = np.random.rand(2, 4)
    Y = my_model(X)

    my_model.info()
    print("##### ----> Modify")
    my_model.wider(added_size=2)
    my_model.deeper()
    my_model.info()
