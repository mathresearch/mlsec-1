from mxnet import gluon, autograd

# Function to define neural network
def custom_model():
    net = gluon.nn.HybridSequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(2048, activation='relu'))
        net.add(gluon.nn.Dense(512, activation='relu'))
        net.add(gluon.nn.Dense(1, activation='sigmoid'))
    return net