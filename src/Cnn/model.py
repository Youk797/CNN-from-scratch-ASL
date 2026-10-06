from .numpy_cupy import xp, _to_numpy
from .utils import softmax
from .layers import Conv2D, ReLU, MaxPool2D, Dense , Flatten


class CNN:

    def __init__(self, num_classes, input_shape=(3, 64, 98)):
        self.num_classes = num_classes
        C, F, T = input_shape 

        self.conv1 = Conv2D(C, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(pool=2, stride=2)

        self.conv2 = Conv2D(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(pool=2, stride=2)
        self.input_shape = input_shape


        self.flat = Flatten()

        x_jetable = xp.zeros((1, C, F, T), dtype=xp.float32)
        z = self.conv1.forward(x_jetable)
        z = self.relu1.forward(z)
        z = self.pool1.forward(z)
        z = self.conv2.forward(z)
        z = self.relu2.forward(z)
        z = self.pool2.forward(z)
        flat_dim = int(z.reshape(1, -1).shape[1])

        self.fc = Dense(flat_dim, num_classes)

        self.layers = [
            self.conv1, self.relu1, self.pool1,
            self.conv2, self.relu2, self.pool2,
            self.flat, self.fc
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x  # logits

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def predict_proba(self, x):
        logits = self.forward(x)
        return softmax(logits, axis=1)

    def params_and_grads(self):
        for layer in self.layers:
            if hasattr(layer, "params_and_grads"):
                yield from layer.params_and_grads()

    def state_dict_numpy(self):
        return {
            "conv1_W": _to_numpy(self.conv1.W),
            "conv1_b": _to_numpy(self.conv1.b),
            "conv2_W": _to_numpy(self.conv2.W),
            "conv2_b": _to_numpy(self.conv2.b),
            "fc_W": _to_numpy(self.fc.W),
            "fc_b": _to_numpy(self.fc.b),
        }

    def load_state_dict_numpy(self, d):
        self.conv1.W = xp.asarray(d["conv1_W"]).astype(xp.float32)
        self.conv1.b = xp.asarray(d["conv1_b"]).astype(xp.float32)
        self.conv2.W = xp.asarray(d["conv2_W"]).astype(xp.float32)
        self.conv2.b = xp.asarray(d["conv2_b"]).astype(xp.float32)
        self.fc.W = xp.asarray(d["fc_W"]).astype(xp.float32)
        self.fc.b = xp.asarray(d["fc_b"]).astype(xp.float32)

