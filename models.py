import tensorflow as tf


class TrackingModel(tf.keras.Model):
    def __init__(self, activation, neurons, output):
        # Calling parent constructor
        super(TrackingModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv1D(
            filters=neurons[0], kernel_size=2, activation=activation[0])
        self.pool = tf.keras.layers.MaxPooling1D()
        self.dense = tf.keras.layers.Dense(units=neurons[1])
        self.dense2 = tf.keras.layers.Dense(units=neurons[2])
        self.dense3 = tf.keras.layers.Dense(units=output)

    def call(self, inputs):
        x = self.conv1(inputs)

        x = self.pool(x)
        x = self.dense(x)
        x = tf.keras.activations.tanh(x)
        x = self.dense2(x)
        x = tf.keras.activations.tanh(x)
        x = self.dense3(x)
        return x


class ConvModel(tf.keras.Model):
    def __init__(self, activation, neurons, output):
        super(ConvModel, self).__init__()
        self.conv0 = tf.keras.layers.Conv1D(
            filters=neurons[0], kernel_size=1, activation=activation[0])
        self.conv1 = tf.keras.layers.Conv1D(
            filters=neurons[1], kernel_size=2, activation=activation[0])
        self.conv2 = tf.keras.layers.Conv1D(
            filters=neurons[2], kernel_size=2, activation=activation[0])
        self.pool = tf.keras.layers.MaxPooling1D()
        self.dense0 = tf.keras.layers.Dense(units=neurons[3])
        self.dense1 = tf.keras.layers.Dense(units=output)

    def call(self, inputs):
        print(inputs)
        x = self.conv0(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dense0()(x)
        x = tf.keras.activations.tanh(x)
        x = self.dens1()(x)
        return x


class TestModel(tf.keras.Model):
    def __init__(self, activation, neurons, output):
        super(TestModel, self). __init__()
        self.conv0 = tf.keras.layers.Conv1D(filters=64, kernel_size=2,
                                            padding='same', activation='tanh')
        self.pool = tf.keras.layers.MaxPooling1D()
        self.conv1 = tf.keras.layers.Conv1D(filters=32, kernel_size=2,
                                            padding='same', activation='tanh')
        self.dense0 = tf.keras.layers.Dense(units=3)

    def call(self, inputs):
        x = self.conv0(inputs)
        x = self.pool(x)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.dense0(x)
        return x

class TestModel2(tf.keras.Model):
    def __init__(self, activation, neurons, output):
        super(TestModel2, self). __init__()
        self.conv0 = tf.keras.layers.Conv1D(filters=64, kernel_size=2,
                                            padding='same', activation='tanh')
        self.pool = tf.keras.layers.MaxPooling1D()
        self.conv1 = tf.keras.layers.Conv1D(filters=32, kernel_size=2,
                                            padding='valid', activation='tanh')
        self.dense0 = tf.keras.layers.Dense(units=3)

    def call(self, inputs):
        x = self.conv0(inputs)
        x = self.pool(x)
        x = self.conv1(x)
        x = self.dense0(x)
        return x

class TestModel3(tf.keras.Model):
    def __init__(self, activation, neurons, output):
        super(TestModel3, self). __init__()
        self.conv0 = tf.keras.layers.Conv1D(filters=64, kernel_size=2,
                                            padding='valid', activation='tanh')
        self.pool = tf.keras.layers.MaxPooling1D()
        self.conv1 = tf.keras.layers.Conv1D(filters=32, kernel_size=2,
                                            padding='valid', activation='tanh')
        self.dense0 = tf.keras.layers.Dense(units=3)

    def call(self, inputs):
        x = self.conv0(inputs)
        x = self.pool(x)
        x = self.dense0(x)
        print(x)
        return x

class TestModel4(tf.keras.Model):
    def __init__(self, activation, neurons, output):
        super(TestModel4, self). __init__()
        self.conv0 = tf.keras.layers.Conv1D(filters=64, kernel_size=2,
                                            padding='valid', activation='tanh')
        self.pool = tf.keras.layers.MaxPooling1D()
        self.conv1 = tf.keras.layers.Conv1D(filters=32, kernel_size=2,
                                            padding='valid', activation='tanh')
        self.dense0 = tf.keras.layers.Dense(units=3)
        self.dense1 = tf.keras.layers.Dense(units=32)
    def call(self, inputs):
        x = self.conv0(inputs)
        x = self.pool(x)
        x = self.dense1(x)
        x = self.dense0(x)
        return x


class EightTrackModelBase(tf.keras.Model):
    def __init__(self, activation, neurons, output):
        super(EightTrackModelBase, self). __init__()
        self.conv0 = tf.keras.layers.Conv1D(filters=64, kernel_size=2,
                                            padding='valid', activation='relu')
        self.pool = tf.keras.layers.MaxPooling1D()
        self.conv1 = tf.keras.layers.Conv1D(filters=32, kernel_size=2,
                                            padding='valid', activation='relu')
        self.conv3 = tf.keras.layers.Conv1D(filters=16, kernel_size=2,
                                            padding='valid', activation='relu')
        self.dense0 = tf.keras.layers.Dense(units=3, activation='relu')
        self.dense1 = tf.keras.layers.Dense(units=32, activation='tanh')
    def call(self, inputs):
        x = self.conv0(inputs)
        x = self.pool(x)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.dense0(x)
        return x

class EightTrackModelRelu(tf.keras.Model):
    def __init__(self, activation, neurons, output):
        super(EightTrackModelRelu, self). __init__()
        self.conv0 = tf.keras.layers.Conv1D(filters=64, kernel_size=2,
                                            padding='valid', activation='relu')
        self.pool = tf.keras.layers.MaxPooling1D()
        self.conv1 = tf.keras.layers.Conv1D(filters=32, kernel_size=2,
                                            padding='valid', activation='relu')
        self.conv3 = tf.keras.layers.Conv1D(filters=16, kernel_size=2,
                                            padding='valid', activation='relu')
        self.dense0 = tf.keras.layers.Dense(units=3, activation='relu')
        self.dense1 = tf.keras.layers.Dense(units=32, activation='relu')
    def call(self, inputs):
        x = self.conv0(inputs)
        x = self.pool(x)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.dense0(x)
        return x
        
class EightTrackModelTanh(tf.keras.Model):
    def __init__(self, activation, neurons, output):
        super(EightTrackModelTanh, self). __init__()
        self.conv0 = tf.keras.layers.Conv1D(filters=64, kernel_size=2,
                                            padding='valid', activation='tanh')
        self.pool = tf.keras.layers.MaxPooling1D()
        self.conv1 = tf.keras.layers.Conv1D(filters=32, kernel_size=2,
                                            padding='valid', activation='tanh')
        self.conv3 = tf.keras.layers.Conv1D(filters=16, kernel_size=2,
                                            padding='valid', activation='tanh')
        self.dense0 = tf.keras.layers.Dense(units=3, activation='tanh')
        self.dense1 = tf.keras.layers.Dense(units=32, activation='tanh')
    def call(self, inputs):
        x = self.conv0(inputs)
        x = self.pool(x)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.dense0(x)
        return x
    
class EightTrackModelAvgPooling(tf.keras.Model):
    def __init__(self, activation, neurons, output):
        super(EightTrackModelAvgPooling, self). __init__()
        self.conv0 = tf.keras.layers.Conv1D(filters=64, kernel_size=2,
                                            padding='valid', activation='tanh')
        self.pool = tf.keras.layers.AveragePooling1D()
        self.conv1 = tf.keras.layers.Conv1D(filters=32, kernel_size=2,
                                            padding='valid', activation='tanh')
        self.conv3 = tf.keras.layers.Conv1D(filters=16, kernel_size=2,
                                            padding='valid', activation='tanh')
        self.dense0 = tf.keras.layers.Dense(units=3, activation='tanh')
        self.dense1 = tf.keras.layers.Dense(units=32, activation='tanh')
    def call(self, inputs):
        x = self.conv0(inputs)
        x = self.pool(x)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.dense0(x)
        return x

class EightTrackModelAvgPooling3Dense(tf.keras.Model):
    def __init__(self, activation, neurons, output):
        super(EightTrackModelAvgPooling3Dense, self). __init__()
        self.conv0 = tf.keras.layers.Conv1D(filters=64, kernel_size=2,
                                            padding='valid', activation='tanh')
        self.pool = tf.keras.layers.AveragePooling1D()
        self.conv1 = tf.keras.layers.Conv1D(filters=32, kernel_size=2,
                                            padding='valid', activation='tanh')
        self.conv3 = tf.keras.layers.Conv1D(filters=16, kernel_size=2,
                                            padding='valid', activation='tanh')
        self.dense0 = tf.keras.layers.Dense(units=3, activation='tanh')
        self.dense1 = tf.keras.layers.Dense(units=64, activation='tanh')
        self.dense2 = tf.keras.layers.Dense(units=32, activation='tanh')
    def call(self, inputs):
        x = self.conv0(inputs)
        x = self.pool(x)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense0(x)
        return x


class TrackModelDropoutRelu(tf.keras.Model):
    def __init__(self, activation, neurons, output, drop_rate):
        super(TrackModelDropoutRelu, self). __init__()
        self.conv0 = tf.keras.layers.Conv1D(filters=64, kernel_size=2,
                                            padding='valid', activation=activation)
        self.pool = tf.keras.layers.MaxPooling1D()
        self.conv1 = tf.keras.layers.Conv1D(filters=32, kernel_size=2,
                                            padding='valid', activation=activation)
        self.conv3 = tf.keras.layers.Conv1D(filters=16, kernel_size=2,
                                            padding='valid', activation=activation)
        self.dense0 = tf.keras.layers.Dense(units=3, activation=activation)
        self.dense1 = tf.keras.layers.Dense(units=32, activation=activation)
        self.dropout = tf.keras.layers.Dropout(drop_rate)
    def call(self, inputs):
        x = self.conv0(inputs)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = self.dense0(x)
        return x
