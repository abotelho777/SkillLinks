import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import *
from sklearn.cross_validation import StratifiedKFold
import time

import DataUtility as du

if __name__ == "__main__":

    # load training and test data
    print "Loading Data..."
    training = []
    tr_label = []
    testing = []
    test_label = []



    # hyper parameters
    autoencoder_units = 200
    num_units = 200
    num_input = 5
    num_output = 3
    dropout = [0.3]

    step_size = 0.01
    batch_size = 10
    num_folds = 10
    num_epochs = 20

    print "Building Network..."
    # autoencoder A->B
    l_input_AB = lasagne.layers.InputLayer(shape=(None, num_input))
    l_dropout_AB = lasagne.layers.DropoutLayer(l_input_AB, p=dropout[0])
    l_hidden_AB = lasagne.layers.DenseLayer(l_dropout_AB, num_units=autoencoder_units)
    l_inverse_AB = lasagne.layers.InverseLayer(l_hidden_AB, l_hidden_AB)
    l_output_AB = lasagne.layers.DenseLayer(l_inverse_AB, num_units=num_input)

    encoder_AB_output = lasagne.layers.get_output(l_output_AB)
    encoder_AB_hidden = lasagne.layers.get_output(l_hidden_AB, deterministic=True)

    # autoencoder B->A
    l_input_BA = lasagne.layers.InputLayer(shape=(None, num_input))
    l_dropout_BA = lasagne.layers.DropoutLayer(l_input_BA, p=dropout[0])
    l_hidden_BA = lasagne.layers.DenseLayer(l_dropout_BA, num_units=autoencoder_units)
    l_inverse_BA = lasagne.layers.InverseLayer(l_hidden_BA, l_hidden_BA)
    l_output_BA = lasagne.layers.DenseLayer(l_inverse_BA, num_units=num_input)

    encoder_BA_output = lasagne.layers.get_output(l_output_BA)
    encoder_BA_hidden = lasagne.layers.get_output(l_hidden_BA, deterministic=True)

    # RBM for link-level features
    l_input = lasagne.layers.InputLayer(shape=(None,autoencoder_units*2))
    l_hidden = lasagne.layers.DenseLayer(l_input,num_units=num_units)
    l_output = lasagne.layers.DenseLayer(l_hidden,num_units=num_output)

    network_output = lasagne.layers.get_output(l_output)

    # cost functions
    target_values = T.matrix('target_output')

    cost_AB = T.nnet.categorical_crossentropy(encoder_AB_output, target_values).mean()
    cost_BA = T.nnet.categorical_crossentropy(encoder_BA_output, target_values).mean()
    cost = T.nnet.categorical_crossentropy(network_output, target_values).mean()

    # parameters
    all_params_AB = lasagne.layers.get_all_params(l_output_AB, trainable=True)
    all_params_BA = lasagne.layers.get_all_params(l_output_BA, trainable=True)
    all_params = lasagne.layers.get_all_params(l_output, trainable=True)

    # updating
    updates_AB = lasagne.updates.adagrad(cost_AB, all_params_AB, step_size)
    updates_BA = lasagne.updates.adagrad(cost_BA, all_params_BA, step_size)
    updates = lasagne.updates.adagrad(cost, all_params, step_size)

    print "Compiling Functions..."
    # autoencoder accessors
    get_hidden_AB = theano.function([l_input_AB.input_var], encoder_AB_hidden, allow_input_downcast=True)
    get_hidden_BA = theano.function([l_input_BA.input_var], encoder_BA_hidden, allow_input_downcast=True)

    get_output_AB = theano.function([l_input_AB.input_var], encoder_AB_output, allow_input_downcast=True)
    get_output_BA = theano.function([l_input_BA.input_var], encoder_BA_output, allow_input_downcast=True)

    # training
    train_AB = theano.function([l_input_AB.input_var, target_values],
                               cost_AB,
                               updates=updates_AB,
                               allow_input_downcast=True)

    train_BA = theano.function([l_input_BA.input_var, target_values],
                               cost_BA,
                               updates=updates_BA,
                               allow_input_downcast=True)

    train = theano.function([l_input.input_var, target_values],
                                    cost,
                                    updates=updates,
                                    allow_input_downcast=True)

    # predicting
    test_AB = theano.function([l_input_AB.input_var, target_values],
                               cost_AB,
                               allow_input_downcast=True)

    test_BA = theano.function([l_input_BA.input_var, target_values],
                               cost_BA,
                               allow_input_downcast=True)

    predict = theano.function([l_input.input_var], network_output, allow_input_downcast=True)
    test = theano.function([l_input.input_var, target_values], cost, allow_input_downcast=True)

    print "\nNumber of Parameters:", count_params(l_output)+(count_params(l_output_AB)*2)

    # functions to feed the autoencoders into the RBM
    def train_network(A_input, B_input, labels):
        AB = np.array(get_hidden_AB(A_input))
        BA = np.array(get_hidden_BA(B_input))
        assert len(AB) == len(BA)
        inputs = np.append(AB,BA,axis=1)
        return train(inputs,labels)


    def test_network(A_input, B_input, labels):
        AB = np.array(get_hidden_AB(A_input))
        BA = np.array(get_hidden_BA(B_input))
        assert len(AB) == len(BA)
        inputs = np.append(AB, BA, axis=1)
        return test(inputs, labels)


    def network_predict(A_input, B_input):
        AB = np.array(get_hidden_AB(A_input))
        BA = np.array(get_hidden_BA(B_input))
        assert len(AB) == len(BA)
        inputs = np.append(AB, BA, axis=1)
        return predict(inputs)


    # distribute into folds
    strat_label = [tr.index(max(tr)) for tr in tr_label]
    skf = StratifiedKFold(strat_label, n_folds=num_folds)

    print "\nTraining AutoEncoders..."

    print "{:<9}".format("  Epoch"), \
        "{:<9}".format("  Train"), \
        "{:<9}".format("  Valid"), \
        "{:<9}".format("  Time"), \
        "\n======================================"
    start_time = time.clock()
    train_err = []
    val_err = []
    # for each epoch...
    for e in range(0, num_epochs):
        epoch_time = time.clock()
        epoch = 0
        eval = 0
        n_train = 0
        n_test = 0

        # train and test
        for ktrain, ktest in skf:
            for i in range(0, len(ktrain), batch_size):
                batch_sample = []
                batch_label = []
                # create a batch of training samples
                for j in range(i, min(len(ktrain), i + batch_size)):
                    batch_sample.append(training[ktrain[j]])
                    batch_label.append(tr_label[ktrain[j]])

                # update and get the cost
                epoch += train_AB(batch_sample,batch_label)
                epoch += train_BA(batch_sample, batch_label)

                n_train += 2

            sample = []
            label = []
            for i in range(0, len(ktest)):
                sample.append(training[ktest[i]])
                label.append(tr_label[ktest[i]])
                n_test += 2

            eval += test_AB(sample, label)
            eval += test_BA(sample, label)

        train_err.append(epoch / n_train)
        val_err.append(eval / n_test)
        print "{:<9}".format("Epoch " + str(e + 1) + ":"), \
            "  {0:.4f}".format(epoch / n_train), \
            "   {0:.4f}".format(eval / n_test), \
            "   {0:.1f}s".format(time.clock() - epoch_time)
    print "Total Training Time:", "{0:.1f}s".format(time.clock() - start_time)

    print "Training Stacked Network..."

    print "{:<9}".format("  Epoch"), \
        "{:<9}".format("  Train"), \
        "{:<9}".format("  Valid"), \
        "{:<9}".format("  Time"), \
        "\n======================================"
    start_time = time.clock()
    train_err = []
    val_err = []
    # for each epoch...
    for e in range(0, num_epochs):
        epoch_time = time.clock()
        epoch = 0
        eval = 0
        n_train = 0
        n_test = 0

        # train and test
        for ktrain, ktest in skf:
            for i in range(0, len(ktrain), batch_size):
                batch_sample = []
                batch_label = []
                # create a batch of training samples
                for j in range(i, min(len(ktrain), i + batch_size)):
                    batch_sample.append(training[ktrain[j]])
                    batch_label.append(tr_label[ktrain[j]])

                # update and get the cost
                epoch += train_AB(batch_sample, batch_label)
                epoch += train_BA(batch_sample, batch_label)

                n_train += 2

            sample = []
            label = []
            for i in range(0, len(ktest)):
                sample.append(training[ktest[i]])
                label.append(tr_label[ktest[i]])
                n_test += 2

            eval += test_AB(sample, label)
            eval += test_BA(sample, label)

        train_err.append(epoch / n_train)
        val_err.append(eval / n_test)
        print "{:<9}".format("Epoch " + str(e + 1) + ":"), \
            "  {0:.4f}".format(epoch / n_train), \
            "   {0:.4f}".format(eval / n_test), \
            "   {0:.1f}s".format(time.clock() - epoch_time)
    print "Total Training Time:", "{0:.1f}s".format(time.clock() - start_time)