import numpy as np
import sys


def sigmoid(x): return 1 / (1 + np.exp(-x))


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def pack(nor_x_train: np.ndarray, y_train: np.ndarray):
    np.warnings.filterwarnings(
        'ignore', category=np.VisibleDeprecationWarning)
    label_list = np.array([[nor_x_train[0], y_train[0]]], dtype=np.ndarray)
    for i in range(len(nor_x_train)):
        label_list = np.append(
            label_list, [[nor_x_train[i], y_train[i]]], 0)
    label_list = np.delete(label_list, 0, 0)
    return label_list


def unpack(x_y_list: np.ndarray):
    if len(x_y_list) == 0:
        return None, None

    x_list = np.array([x_y_list[0][0]])
    y_list = np.array([x_y_list[0][1]])
    for i in x_y_list:
        x_list = np.append(x_list, [i[0]], 0)
        y_list = np.append(y_list, [i[1]], 0)

    x_list = np.delete(x_list, 0, 0)
    y_list = np.delete(y_list, 0, 0)
    return x_list, y_list


def decode(encoded: np.ndarray):
    dec = np.array([])

    for e in encoded:
        dec = np.append(dec, [i for i in range(len(e)) if e[i]])
    return dec.astype(int)


def accuracy_epoch(params: dict, x_y_list: np.ndarray):
    y_test = []
    x_list, y_list = unpack(x_y_list)

    for i in x_list:
        h2 = fprop2(i.reshape((784, 1)), params)
        y_test = np.append(y_test, [np.argmax(h2)])

    b = (decode(y_list) == y_test)
    return np.count_nonzero(b == 1) / len(y_test)


def fprop2(x, params: dict):
    # Follows procedure given in notes
    W1, b1, W2, b2 = params.values()
    z1 = np.dot(W1, x) + b1
    h1 = sigmoid(z1)
    z2 = np.dot(W2, h1) + b2
    return softmax(z2)


def fprop(x, y, params: dict):
    # Follows procedure given in notes
    W1, b1, W2, b2 = params.values()
    z1 = np.dot(W1, x) + b1
    h1 = sigmoid(z1)
    z2 = np.dot(W2, h1) + b2
    h2 = softmax(z2)
    loss = -y * np.log(h2)
    ret = {'x': x, 'y': y, 'z1': z1, 'h1': h1,
           'z2': z2, 'h2': h2, 'loss': loss}
    for key in params:
        ret[key] = params[key]
    return ret


def bprop(fprop_cache: dict, rate: float):
    # Follows procedure given in notes
    x, y, z1, h1, z2, h2, loss = [fprop_cache[key] for key in ('x', 'y', 'z1', 'h1',
                                                               'z2', 'h2', 'loss')]
    dz2 = (h2 - y)                                          # dL/dz2
    dW2 = np.dot(dz2, h1.T)                                 # dL/dz2 * dz2/dw2
    db2 = dz2                                               # dL/dz2 * dz2/db2
    dz1 = np.dot(fprop_cache['W2'].T,
                 (h2 - y)) * sigmoid(z1) * (1-sigmoid(z1))  # dL/dz2 * dz2/dh1 * dh1/dz1
    # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1

    dW1 = np.dot(dz1, x.T)
    # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/db1
    db1 = dz1

    return {'b1': rate * db1, 'W1': rate * dW1, 'b2': rate * db2, 'W2': rate * dW2}


def neuralNetwork(label_list: np.ndarray, x_test: np.ndarray, epoch: int, rate: float, lines: int):
    W1 = (np.random.rand(lines, 784) - 0.5) / (784 * lines) ** 0.5
    b1 = (np.random.rand(lines, 1) - 0.5) / lines ** 0.5
    W2 = (np.random.rand(10, lines) - 0.5) / (10 * lines) ** 0.5
    b2 = (np.random.rand(10, 1) - 0.5) / lines ** 0.5
    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    # training
    for e in range(epoch):
        for label in label_list:
            fprop_cache = fprop(label[0].reshape(
                (784, 1)), label[1].reshape((10, 1)), params)
            bprop_cache = bprop(fprop_cache, rate)

            # updating weights nd bias
            for key in bprop_cache:
                params[key] -= bprop_cache[key]

    y_test = np.array([])
    for i in x_test:
        h2 = fprop2(i.reshape((784, 1)), params)
        y_test = np.append(y_test, [np.argmax(h2)])

    return y_test.astype(int)


if __name__ == "__main__":
    train_x_fname, train_y_fname, test_x_fname = sys.argv[1:]
    out_fname = 'test_y'

    x_train = np.loadtxt(train_x_fname).astype(np.longdouble)
    x_test = np.loadtxt(test_x_fname).astype(np.longdouble)
    y_train = np.loadtxt(train_y_fname).astype(np.longdouble)
    y_train = y_train.astype(int)

    # normalization
    nor_x_train = x_train / 255
    nor_x_test = x_test / 255

    # one hot encoding for y train
    y_train_enc = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    for i in range(len(y_train)):
        y_train_enc = np.append(
            y_train_enc, [np.array([y_train[i] == k for k in range(10)])], 0)
    y_train_enc = np.delete(y_train_enc, 0, 0)

    label_list = pack(nor_x_train, y_train_enc)
    label_copy = label_list.copy()
    np.random.shuffle(label_copy)
    outFile = None
    try:
        outFile = open(out_fname, "w")
        line = 128
        y_test = np.loadtxt("test_labels.txt").astype(np.longdouble)
        accuracy_average = 0.
        y_test_dec = neuralNetwork(label_copy, x_test, 6, 0.05, line)
        accuracy_average += np.count_nonzero(y_test_dec ==
                                             y_test) / len(y_test)

        for i in y_test_dec:
            outFile.write(i.__str__() + '\n')

    except Exception as e:
        pass
        # print(e)   shoulb be removed when submmitting

    finally:
        if outFile:
            outFile.close()
