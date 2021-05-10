from uwnet import *


def conv_net():
    l = [
        make_convolutional_layer(32, 32, 3, 8, 3, 1),
        make_activation_layer(RELU),
        make_maxpool_layer(32, 32, 8, 3, 2),
        make_convolutional_layer(16, 16, 8, 16, 3, 1),
        make_activation_layer(RELU),
        make_maxpool_layer(16, 16, 16, 3, 2),
        make_convolutional_layer(8, 8, 16, 32, 3, 1),
        make_activation_layer(RELU),
        make_maxpool_layer(8, 8, 32, 3, 2),
        make_convolutional_layer(4, 4, 32, 64, 3, 1),
        make_activation_layer(RELU),
        make_maxpool_layer(4, 4, 64, 3, 2),
        make_connected_layer(256, 10),
        make_activation_layer(SOFTMAX),
    ]
    print("using conv_net")
    return make_net(l)


# def conv_net():
#     l = [
#         make_convolutional_layer(32, 32, 3, 8, 3, 2),
#         make_activation_layer(RELU),
#         make_convolutional_layer(16, 16, 8, 16, 3, 2),
#         make_activation_layer(RELU),
#         make_convolutional_layer(8, 8, 16, 32, 3, 2),
#         make_activation_layer(RELU),
#         make_convolutional_layer(4, 4, 32, 64, 3, 2),
#         make_activation_layer(RELU),
#         make_connected_layer(256, 10),
#         make_activation_layer(SOFTMAX),
#     ]
#     return make_net(l)


def conn_net():
    l = [
        make_connected_layer(32 * 32 * 3, 256),
        make_activation_layer(RELU),
        make_connected_layer(256, 128),
        make_activation_layer(RELU),
        make_connected_layer(128, 256),
        make_activation_layer(RELU),
        make_connected_layer(256, 450),
        make_activation_layer(RELU),
        make_connected_layer(450, 256),
        make_activation_layer(RELU),
        make_connected_layer(256, 128),
        make_activation_layer(RELU),
        make_connected_layer(128, 10),
        make_activation_layer(SOFTMAX),
    ]
    print("using conn_net")
    return make_net(l)


print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test = load_image_classification_data("cifar/cifar.test", "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 5000
rate = 0.01
momentum = 0.9
decay = 0.005

m = conn_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# How accurate is the fully connected network vs the convnet when they use similar number of operations?
# Why are you seeing these results? Speculate based on the information you've gathered and what you know about DL and ML.
# Your answer:
#
# The convnet makes 8 * 28 * 1024 + 16 * 72 * 16^2 + 32 * 144 * 64
# + 64 * 288 * 16 + 10 * 256 * 1 = 1108480 operations.
#
# The fully connected network makes 3072*256 + 256*128 + 128*256
# + 256*450 + 450*256 + 256*128 + 128*10 = 1116416 operations.
#
# convnet training accuracy:         %f 0.6713799834251404
# convnet test accuracy:             %f 0.6280999779701233
# fully connected training accuracy: %f 0.5281400084495544
# fully connected test accuracy:     %f 0.48809999227523804
#
# I suspect the convnet performs better due to its ability to process spatial information about each image.
# In the fully connected network, every neuron in the first layer is connected to every neuron in the
# second layer, every neuron in the second layer is connected to every neuron in the third layer, etc. This
# causes spatial information contained within the image to be lost, as each pixel is considered equally related
# to every other pixel, regardless of the distance between them. On the other hand, the use of convolutions
# in the convnet causes the connections in the network to be sparser. Pixels that are close to each other
# are considered more related to one another than pixels far away from each other, since we are using
# weighted sums of small areas in each image.
# Since spatial information is so important to understanding what an image depicts, the fact that the convnet
# can use this information causes it to outperform the fully connected network.
