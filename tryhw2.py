from uwnet import *


def conv_net():
    l = [
        make_convolutional_layer(32, 32, 3, 8, 3, 2),
        make_activation_layer(RELU),
        make_maxpool_layer(16, 16, 8, 3, 2),
        make_convolutional_layer(8, 8, 8, 16, 3, 1),
        make_activation_layer(RELU),
        make_maxpool_layer(8, 8, 16, 3, 2),
        make_convolutional_layer(4, 4, 16, 32, 3, 1),
        make_activation_layer(RELU),
        make_connected_layer(512, 10),
        make_activation_layer(SOFTMAX),
    ]
    # l = [
    #     make_convolutional_layer(32, 32, 3, 8, 3, 2),
    #     make_batchnorm_layer(8),
    #     make_activation_layer(RELU),
    #     make_maxpool_layer(16, 16, 8, 3, 2),
    #     make_convolutional_layer(8, 8, 8, 16, 3, 1),
    #     make_batchnorm_layer(16),
    #     make_activation_layer(RELU),
    #     make_maxpool_layer(8, 8, 16, 3, 2),
    #     make_convolutional_layer(4, 4, 16, 32, 3, 1),
    #     make_batchnorm_layer(32),
    #     make_activation_layer(RELU),
    #     make_connected_layer(512, 10),
    #     make_activation_layer(SOFTMAX),
    # ]
    return make_net(l)


print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test = load_image_classification_data("cifar/cifar.test", "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
# iters = 500
# rate = 0.01
iters = 300
rate = 0.1
momentum = 0.9
decay = 0.005

m = conv_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
# anneal
iters = 50
rate = 0.05
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
rate = 0.025
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
rate = 0.0125
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
rate = 0.01
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# 7.6 Question: What do you notice about training the convnet with/without batch normalization? How does it affect convergence? How does it affect what magnitude of learning rate you can use?
# Write down any observations from your experiments:
# Adding batch normalization layers significantly improved the network performance. Using the default hyperparameters, the network without batch norm had a test accuracy
# of 40.5%, whereas the network with batch norm had an accuracy of 52.9%.
#
# With batch normalization, I was able to improve the network performance by increasing the initial learning rate to 0.1. The batch norm network with a learning rate of 0.01 had test
# accuracy of 52.9%, whereas the network with an initial learning rate of 0.1 and annealing had test accuracy of 54.1%.
# On the other hand, increasing the learning rate diminished the performance of the non-batch norm network. Increasing its learning rate decreased its test accuracy from
# 40.5% to 39.8%.
#
# Convergence of the non-batch norm network seems to plateau after the learning rate is diminished. During the last 200 iterations, the loss went from 1.70 to 1.66.
# On the other hand, the batch norm network seems to continue learning even as the learning rate decreases. During its last 200 iterations, the loss went from 1.55 to 1.16.
