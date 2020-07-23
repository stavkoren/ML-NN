import sys

import numpy as np
import scipy.special as scipy

FIRST_LAYER = 28 * 28
HIDDEN_LAYER = 128
OUTPUT_LAYER = 10
UNIFORM_UPPER_BOUNDARY = 0.08
UNIFORM_LOWER_BOUNDARY = -0.08
ETA = 0.1
EPCHOS = 50



class Network_Parameters:
    def __init__(self):
        self.w1,self.w2,self.b1,self.b2=[self.init_network()[key] for key in ('w1', 'w2', 'b1', 'b2')]

    # initialize weights and biases
    def init_network(self):
        w1 = np.random.uniform(UNIFORM_LOWER_BOUNDARY, UNIFORM_UPPER_BOUNDARY, (HIDDEN_LAYER, FIRST_LAYER))
        w2 = np.random.uniform(UNIFORM_LOWER_BOUNDARY, UNIFORM_UPPER_BOUNDARY, (OUTPUT_LAYER, HIDDEN_LAYER))
        b1 = np.random.uniform(UNIFORM_LOWER_BOUNDARY, UNIFORM_UPPER_BOUNDARY, HIDDEN_LAYER)
        b2 = np.random.uniform(UNIFORM_LOWER_BOUNDARY, UNIFORM_UPPER_BOUNDARY, OUTPUT_LAYER)
        return {'w1': w1, 'w2': w2, 'b1': b1, 'b2': b2}



def calculate_z(x, w, b):
    return np.dot(w, x) + b


def ReLu(z):
    return np.maximum(0, z)


def loss(v2,y):
    return np.negative(np.log(v2[y]))


def normalize_values(x):
    if x.max() != 0:
        return x / x.max()
    return x


def forward_propagation(network_params, x, y=-1):
    z1 = calculate_z(x, network_params.w1, network_params.b1)
    v1 = normalize_values(ReLu(z1))
    z2 = calculate_z(v1, network_params.w2, network_params.b2)
    v2 = scipy.softmax(z2)
    l=loss(v2,y)
    ret = {'v1': v1, 'v2': v2, 'z1': z1, 'z2': z2, 'loss': l}

    return ret


def reLu_derivative(x):
    x[x > 0] = 1
    x[x <= 0] = 0
    return x


def backward_propagation(x, y, forword_params, network_params):
    v1, v2, z1, z2 = [forword_params[key] for key in ('v1', 'v2', 'z1', 'z2')]
    e1 = v2
    e1[y] = e1[y] - 1
    g_l = np.reshape(e1, (OUTPUT_LAYER, 1))
    delta_b2 = g_l
    w2_transpose = np.transpose(network_params.w2)
    v1_transpose = np.reshape(np.transpose(v1), (1, HIDDEN_LAYER))
    delta_w2 = np.dot(g_l, v1_transpose)
    delta_b1 = np.dot(w2_transpose, e1)
    delta_b1 *= reLu_derivative(z1)
    x_tag = np.reshape(x, (1, FIRST_LAYER))
    delta_w1 = np.dot(np.reshape(delta_b1, (HIDDEN_LAYER, 1)), x_tag)
    return {'delta_w1': delta_w1, 'delta_w2': delta_w2, 'delta_b1': delta_b1, 'delta_b2': delta_b2}


def update_weights_and_biases(backword_params, network_params):
    delta_w1, delta_w2, delta_b1, delta_b2 = [backword_params[key] for key in (
        'delta_w1', 'delta_w2', 'delta_b1', 'delta_b2')]
    network_params.w1 -= delta_w1 * ETA
    network_params.w2 -= delta_w2 * ETA
    network_params.b1 -= delta_b1 * ETA
    network_params.b2 -= np.reshape(delta_b2, OUTPUT_LAYER) * ETA
    return network_params


def check_success_rate(predict_x, predict_y, network_params):
    count_correct_predictions = 0
    for x, y in zip(predict_x, predict_y):
        v1, v2, z1, y_hat, l = [forward_propagation(network_params, x, y)
                                [key] for key in ('v1', 'v2', 'z1', 'z2', 'loss')]
        if y == y_hat.argmax(axis=0):
            count_correct_predictions += 1
    precent = (count_correct_predictions / len(predict_y)) * 100
    print('success rate: ', precent)
    return precent


def train_neural_network(train_x, train_y, predict_x, predict_y):
    network_parms = Network_Parameters()
    for i in range(EPCHOS):
        for x, y in zip(train_x, train_y):
            forward_params = forward_propagation(network_parms, x, y)
            backward_params = backward_propagation(x, y, forward_params, network_parms)
            # update weights
            v1, v2, z1, y_hat, l = [forward_params
                                    [key] for key in ('v1', 'v2', 'z1', 'z2', 'loss')]
            if l>0:
                network_parms = update_weights_and_biases(backward_params, network_parms)
        check_success_rate(predict_x, predict_y, network_parms)
    return network_parms


def write_predictions(test_x, network_params):
    file = open("test_y", "w")
    for x in test_x:
        v1, v2, z1, z2 = [forward_propagation(network_params, x)
                               [key] for key in ('v1', 'v2', 'z1', 'z2')]
        file.write(str(v2.argmax()) + '\n')
    file.close()




train_x, train_y, test_x = sys.argv[1], sys.argv[2], sys.argv[3]
train_x = np.loadtxt(train_x)
train_y = np.loadtxt(train_y, dtype=np.int32)
test_x = np.loadtxt(test_x)
neural_params=train_neural_network(train_x[:int(len(train_y) * 0.8), :], train_y[:int(len(train_y) * 0.8)],
               train_x[int(len(train_y) * 0.8):, :], train_y[int(len(train_y) * 0.8):])
write_predictions(test_x,neural_params)



