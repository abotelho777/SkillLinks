import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import *
from sklearn.cross_validation import StratifiedKFold
import time

import DataUtility as du


class AutoEncoder:

    def __init__(self, num_hidden):
        self.training = []
        self.num_units = num_hidden
        self.num_input = 0
        self.num_output = 0
        self.step_size = 0.01
        self.batch_size = 10
        self.num_folds = 10
        self.num_epochs = 20
        self.dropout = 0

        self.l_in = None
        self.l_drop = None
        self.l_hidden = None
        self.l_inverse = None
        self.l_output = None

        self.target_values = T.matrix('target_output')

        self.network_output = None
        self.hidden_output = None
        self.cost = None
        self.all_params = None
        self.updates = None
        self.get_output = None
        self.get_hidden = None
        self.predict_network = None
        self.train_network = None
        self.test_network = None
        self.network_output_nodrop = None
        self.cost_nodrop = None

        self.train_validation = [['Training Error'],['Validation Error']]

        self.isBuilt = False
        self.isTrained = False
        self.isInitialized = False

    def set_hyperparams(self, step_size=.01, dropout=0.0, batch_size=10, num_epochs=20, num_folds=10):
        self.step_size = step_size
        self.dropout = dropout
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_folds = num_folds
        self.isBuilt=False

    def set_training_params(self,batch_size,num_epochs,num_folds=10):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_folds = num_folds

    def build_network(self):
        print "\nBuilding Network..."

        if not self.isInitialized:
            self.l_in = lasagne.layers.InputLayer(shape=(None, self.num_input))
            # self.l_drop = lasagne.layers.DropoutLayer(self.l_in, p=self.dropout)
            self.l_hidden = lasagne.layers.DenseLayer(self.l_in, num_units=self.num_units,
                                                      W=lasagne.init.Normal(),
                                                      nonlinearity=lasagne.nonlinearities.sigmoid)
            self.l_inverse = lasagne.layers.InverseLayer(self.l_hidden, self.l_hidden)
            self.l_output = lasagne.layers.DenseLayer(self.l_inverse, num_units=self.num_output)
            self.isInitialized = True

        self.network_output = lasagne.layers.get_output(self.l_output)
        self.network_output_nodrop = lasagne.layers.get_output(self.l_output, deterministic=True)
        self.hidden_output = lasagne.layers.get_output(self.l_hidden, deterministic=True)

        self.cost = lasagne.objectives.squared_error(self.network_output, self.target_values).mean().sqrt()
        self.cost_nodrop = lasagne.objectives.squared_error(self.network_output_nodrop,self.target_values).mean().sqrt()
        #self.cost = T.nnet.categorical_crossentropy(self.network_output, self.target_values).mean()
        self.all_params = lasagne.layers.get_all_params(self.l_output, trainable=True)
        self.updates = lasagne.updates.adagrad(self.cost, self.all_params, self.step_size)

        self.get_hidden = theano.function([self.l_in.input_var], self.hidden_output, allow_input_downcast=True)
        self.get_output = theano.function([self.l_in.input_var], self.network_output, allow_input_downcast=True)

        self.train_network = theano.function([self.l_in.input_var, self.target_values], self.cost, updates=self.updates,
                                             allow_input_downcast=True)
        self.predict_network = theano.function([self.l_in.input_var], self.network_output_nodrop, allow_input_downcast=True)
        self.test_network = theano.function([self.l_in.input_var, self.target_values], self.cost_nodrop,
                                            allow_input_downcast=True)

        self.isBuilt = True
        print "Network Params:", count_params(self.l_output)

    def train(self, training, output=None):
        if output is None:
            output = training

        assert len(training) == len(output)

        self.num_input = du.len_deepest(training)
        self.num_output = du.len_deepest(output)

        training = du.transpose(training)
        output = du.transpose(output)

        for i in range(0,len(training)):
            training[i] = du.normalize(training[i])
            output[i] = du.normalize(output[i])

        training = du.transpose(training)
        output = du.transpose(output)

        if not self.isBuilt:
            self.build_network()

        print "Input Nodes:", self.num_input
        print "Output Nodes:", self.num_output

        # introduce cross-validation
        from sklearn.cross_validation import StratifiedKFold

        strat_label = []
        for i in range(0, len(training)):
            strat_label.append(1)

        skf = StratifiedKFold(strat_label, n_folds=self.num_folds)

        print"Number of Folds:", len(skf)

        print "Training Samples:", len(training)

        print("\nTraining AutoEncoder...")
        print "{:<9}".format("  Epoch"), \
            "{:<9}".format("  Train"), \
            "{:<9}".format("  Valid"), \
            "{:<9}".format("  Time"), \
            "\n======================================"
        start_time = time.clock()
        train_err = []
        val_err = []
        # for each epoch...
        for e in range(0, self.num_epochs):
            epoch_time = time.clock()
            epoch = 0
            eval = 0
            n_train = 0
            n_test = 0

            # train and test
            for ktrain, ktest in skf:
                for i in range(0, len(ktrain), self.batch_size):
                    batch_sample = []
                    batch_label = []
                    # create a batch of training samples
                    for j in range(i, min(len(ktrain), i + self.batch_size)):
                        #print training[ktrain[j]]
                        batch_sample.append(training[ktrain[j]])
                        batch_label.append(output[ktrain[j]])

                    # update and get the cost
                    #print batch_sample
                    #print self.get_output(batch_sample)
                    #print batch_label
                    epoch += self.train_network(batch_sample, batch_label)

                    n_train += 1

                sample = []
                label = []
                for i in range(0, len(ktest)):
                    sample.append(training[ktest[i]])
                    label.append(output[ktest[i]])
                n_test += 1
                eval += self.test_network(sample, label)

            train_err.append(epoch / n_train)
            val_err.append(eval / n_test)
            print "{:<11}".format("Epoch " + str(e + 1) + ":"), \
                "{:<9}".format("{0:.4f}".format(epoch / n_train)), \
                "{:<9}".format("{0:.4f}".format(eval / n_test)), \
                "{:<9}".format("{0:.1f}s".format(time.clock() - epoch_time))
        print "Total Training Time:", "{0:.1f}s".format(time.clock() - start_time)

    def predict(self, samples):
        return self.predict_network(samples)

    def get_hidden_layer(self, samples):
        return self.get_hidden(samples)

class FFNNet:
    f_predictions = T.fvector('func_predictions')
    f_labels = T.ivector('func_labels')
    ACE_cost = T.nnet.categorical_crossentropy(f_predictions, f_labels).mean()
    AverageCrossEntropy = theano.function([f_predictions, f_labels], ACE_cost,
                                          allow_input_downcast=True)

    def __init__(self, num_hidden):
        self.training = []
        self.num_units = num_hidden
        self.num_input = 0
        self.num_output = 0
        self.step_size = 0.01
        self.batch_size = 10
        self.num_folds = 10
        self.num_epochs = 20
        self.dropout = 0

        self.l_in = None
        self.l_drop = None
        self.l_hidden = None
        self.l_hidden2 = None
        self.l_output = None

        self.target_values = T.matrix('target_output')

        self.network_output = None
        self.hidden_output = None
        self.cost = None
        self.all_params = None
        self.updates = None
        self.get_output = None
        self.get_hidden = None
        self.predict_network = None
        self.train_network = None
        self.test_network = None
        self.network_output_nodrop = None
        self.cost_nodrop = None

        self.train_validation = [['Training Error'],['Validation Error']]

        self.isBuilt = False
        self.isTrained = False
        self.isInitialized = False

    def set_hyperparams(self, step_size=.01, dropout=0.0, batch_size=10, num_epochs=20, num_folds=10):
        self.step_size = step_size
        self.dropout = dropout
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_folds = num_folds
        self.isBuilt=False

    def set_training_params(self,batch_size,num_epochs,num_folds=10):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_folds = num_folds

    def build_network(self):
        print "\nBuilding Network..."

        if not self.isInitialized:
            self.l_in = lasagne.layers.InputLayer(shape=(None, self.num_input))
            self.l_drop = lasagne.layers.DropoutLayer(self.l_in, p=self.dropout)
            self.l_hidden = lasagne.layers.DenseLayer(self.l_drop, num_units=self.num_units,
                                                      W=lasagne.init.Normal(),
                                                      nonlinearity=lasagne.nonlinearities.sigmoid)
            self.l_hidden2 = lasagne.layers.DenseLayer(self.l_hidden, num_units=self.num_units,
                                                      W=lasagne.init.Normal(),
                                                      nonlinearity=lasagne.nonlinearities.sigmoid)
            self.l_output = lasagne.layers.DenseLayer(self.l_hidden, num_units=self.num_output,
                                                      nonlinearity=lasagne.nonlinearities.softmax)
            self.isInitialized = True

        self.network_output = lasagne.layers.get_output(self.l_output)
        self.network_output_nodrop = lasagne.layers.get_output(self.l_output, deterministic=True)
        self.hidden_output = lasagne.layers.get_output(self.l_hidden, deterministic=True)

        self.cost = lasagne.objectives.categorical_crossentropy(self.network_output,self.target_values).mean()
        self.cost_nodrop = lasagne.objectives.categorical_crossentropy(self.network_output_nodrop,
                                                                       self.target_values).mean()
        #self.cost = T.nnet.categorical_crossentropy(self.network_output, self.target_values).mean()
        self.all_params = lasagne.layers.get_all_params(self.l_output, trainable=True)
        self.updates = lasagne.updates.adagrad(self.cost, self.all_params, self.step_size)

        self.get_hidden = theano.function([self.l_in.input_var], self.hidden_output, allow_input_downcast=True)
        self.get_output = theano.function([self.l_in.input_var], self.network_output, allow_input_downcast=True)

        self.train_network = theano.function([self.l_in.input_var, self.target_values], self.cost, updates=self.updates,
                                             allow_input_downcast=True)
        self.predict_network = theano.function([self.l_in.input_var], self.network_output_nodrop, allow_input_downcast=True)
        self.test_network = theano.function([self.l_in.input_var, self.target_values], self.cost_nodrop,
                                            allow_input_downcast=True)

        self.isBuilt = True
        print "Network Params:", count_params(self.l_output)

    def train(self, training, output):

        self.num_input = du.len_deepest(training)
        self.num_output = du.len_deepest(output)

        training = du.transpose(training)
        output = du.transpose(output)

        for i in range(0,len(training)):
            training[i] = du.normalize(training[i])

        training = du.transpose(training)
        output = du.transpose(output)

        if not self.isBuilt:
            self.build_network()

        print "Input Nodes:", self.num_input
        print "Output Nodes:", self.num_output

        # introduce cross-validation
        from sklearn.cross_validation import StratifiedKFold

        strat_label = []
        for i in range(0, len(training)):
            strat_label.append(output[i].index(max(output[i])))

        skf = StratifiedKFold(strat_label, n_folds=self.num_folds)

        print"Number of Folds:", len(skf)

        print "Training Samples:", len(training)

        print("\nTraining Feed-Forward Network...")
        print "{:<9}".format("  Epoch"), \
            "{:<9}".format("  Train"), \
            "{:<9}".format("  Valid"), \
            "{:<9}".format("  Time"), \
            "\n======================================"
        start_time = time.clock()
        train_err = []
        val_err = []
        # for each epoch...
        for e in range(0, self.num_epochs):
            epoch_time = time.clock()
            epoch = 0
            eval = 0
            n_train = 0
            n_test = 0

            # train and test
            for ktrain, ktest in skf:
                for i in range(0, len(ktrain), self.batch_size):
                    batch_sample = []
                    batch_label = []
                    # create a batch of training samples
                    for j in range(i, min(len(ktrain), i + self.batch_size)):
                        #print training[ktrain[j]]
                        batch_sample.append(training[ktrain[j]])
                        batch_label.append(output[ktrain[j]])

                    # update and get the cost
                    #print batch_sample
                    #print self.get_output(batch_sample)
                    #print batch_label
                    epoch += self.train_network(batch_sample, batch_label)

                    n_train += 1

                sample = []
                label = []
                for i in range(0, len(ktest)):
                    sample.append(training[ktest[i]])
                    label.append(output[ktest[i]])
                n_test += 1
                eval += self.test_network(sample, label)

            train_err.append(epoch / n_train)
            val_err.append(eval / n_test)
            print "{:<11}".format("Epoch " + str(e + 1) + ":"), \
                "{:<9}".format("{0:.4f}".format(epoch / n_train)), \
                "{:<9}".format("{0:.4f}".format(eval / n_test)), \
                "{:<9}".format("{0:.1f}s".format(time.clock() - epoch_time))
        print "Total Training Time:", "{0:.1f}s".format(time.clock() - start_time)

    def test(self, samples, test_labels,label_names=None):
        # test each using held-out data
        test = samples

        # if test_labels is None:
        #     return self.predict(test_samples)

        label_test = test_labels
        print("\nTesting...")
        print "Test Samples:", len(test)

        classes = []
        p_count = 0

        avg_class_err = []
        avg_err = self.test_network(test, label_test)

        predictions = self.predict_network(test)

        for i in range(0, len(label_test)):
            p_count += 1
            classes.append(label_test[i].tolist())


        predictions = np.round(predictions, 3).tolist()

        actual = []
        pred = []
        cor = []

        # get the percent correct for the predictions
        # how often the prediction is right when it is made
        for i in range(0, len(predictions)):
            c = classes[i].index(max(classes[i]))
            actual.append(c)

            p = predictions[i].index(max(predictions[i]))
            pred.append(p)
            cor.append(int(c == p))

        # calculate a naive unfair baseline using averages
        avg_class_pred = np.mean(label_test, 0)

        print "Predicting:", avg_class_pred, "for baseline*"
        for i in range(0, len(label_test)):
            res = FFNNet.AverageCrossEntropy(np.array(avg_class_pred), np.array(classes[i]))
            avg_class_err.append(res)
            # res = RNN_GRU.AverageCrossEntropy(np.array(predictions_GRU[i]), np.array(classes[i]))
            # avg_err_GRU.append(res)
        print "*This is calculated from the TEST labels"

        from sklearn.metrics import roc_auc_score, f1_score
        from skll.metrics import kappa

        kpa = []
        auc = []
        f1s = []
        t_pred = du.transpose(predictions)
        t_lab = du.transpose(label_test)

        for i in range(0, len(t_lab)):
            # if i == 0 or i == 3:
            #    t_pred[i] = du.normalize(t_pred[i],method='max')
            kpa.append(kappa(t_lab[i], t_pred[i]))
            auc.append(roc_auc_score(t_lab[i], t_pred[i]))
            temp_p = [round(j) for j in t_pred[i]]
            if np.nanmax(temp_p) == 0:
                f1s.append(0)
            else:
                f1s.append(f1_score(t_lab[i], temp_p))

        print "\nBaseline Average Cross-Entropy:", "{0:.4f}".format(np.nanmean(avg_class_err))
        print "\nNetwork Performance:"
        print "Average Cross-Entropy:", "{0:.4f}".format(np.nanmean(avg_err))
        print "AUC:", "{0:.4f}".format(np.nanmean(auc))
        print "Kappa:", "{0:.4f}".format(np.nanmean(kpa))
        print "F1 Score:", "{0:.4f}".format(np.nanmean(f1s))
        print "Percent Correct:", "{0:.2f}%".format(np.nanmean(cor) * 100)

        print "\n{:<15}".format("  Label"), \
            "{:<9}".format("  AUC"), \
            "{:<9}".format("  Kappa"), \
            "{:<9}".format("  F Stat"), \
            "\n=============================================="

        if label_names is None or len(label_names) != len(t_lab):
            label_names = []
            for i in range(0, len(t_lab)):
                label_names.append("Label " + str(i + 1))

        for i in range(0, len(t_lab)):
            print "{:<15}".format(label_names[i]), \
                "{:<9}".format("  {0:.4f}".format(auc[i])), \
                "{:<9}".format("  {0:.4f}".format(kpa[i])), \
                "{:<9}".format("  {0:.4f}".format(f1s[i]))
        print "\n=============================================="
        actual = []
        predicted = []
        for i in range(0, len(predictions)):
            actual.append(label_test[i].tolist().index(max(label_test[i])))
            predicted.append(predictions[i].index(max(predictions[i])))

        from sklearn.metrics import confusion_matrix
        print confusion_matrix(actual, predicted)

        return predictions

# adds redundancy for students
def add_representation(data,labels,label_column,duplicate=10,threshold=0.0):
    assert len(data) == len(labels)
    print "Adding Representation to label:",label_column
    ndata = []
    nlabel = []
    for i in range(0,len(data)):
        represent = 1
        if labels[i] is list:
            if np.nanmean(labels[i], 0)[label_column] > threshold:
                represent = duplicate
        else:
            if labels[i][label_column] > threshold:
                represent = duplicate

        for j in range(0,represent):
            ndata.append(data[i])
            nlabel.append(labels[i])

    ndata,nlabel = du.shuffle(ndata,nlabel)
    return np.array(ndata),np.array(nlabel)

def load_skill_data(data_filename, prereq_file, nolink_file):
    print "Loading Data..."
    data, headers = du.loadCSVwithHeaders(data_filename)
    prereqs = du.loadCSV(prereq_file)
    nolink = du.loadCSV(nolink_file)

    samples = []
    labels = []

    for i in range(0, len(headers)):
        print '{:>2}:  {:<18} {:<12}'.format(str(i), headers[i], data[0][i])

    print "Hierarchy Structure:"
    for p in prereqs:
        print p[0], '->', p[1]

    students = du.unique(du.transpose(data)[2])
    for i in range(0, len(students)):
        student_set = du.select(data, students[i], '==', 2)

        for p in prereqs:
            post = du.select(student_set, p[1], '==', 0)
            if not len(post) == 0:

                post = post[0]
                pre = du.select(du.select(student_set, p[0], '==', 0), post[1], '<', 1)
                rem = du.select(du.select(student_set, p[0], '==', 0), post[1], '>', 1)

                if not (len(pre) == 0 or len(rem) == 0):
                    pre = pre[0]
                    rem = rem[0]

                    samp_pre = []
                    samp_post = []
                    samp_rem = []
                    for j in range(3, 8):
                        samp_pre.append(pre[j])
                        samp_post.append(post[j])
                        samp_rem.append(rem[j])

                    samples.append([samp_pre, samp_post, samp_rem, [p[0], p[1]]])
                    labels.append([1, 0, 0])

                    # print pre
                    # print post
                    # print rem
                    # print ' '

            post = du.select(student_set, p[0], '==', 0)
            if not len(post) == 0:
                post = post[-1]
                pre = du.select(du.select(student_set, p[1], '==', 0), post[1], '<', 1)
                rem = du.select(du.select(student_set, p[1], '==', 0), post[1], '>', 1)

                if not (len(pre) == 0 or len(rem) == 0):
                    pre = pre[0]
                    rem = rem[0]

                    samp_pre = []
                    samp_post = []
                    samp_rem = []
                    for j in range(3, 8):
                        samp_pre.append(pre[j])
                        samp_post.append(post[j])
                        samp_rem.append(rem[j])

                    samples.append([samp_pre, samp_post, samp_rem, [p[1], p[0]]])
                    labels.append([0, 0, 1])

                    # print pre
                    # print post
                    # print rem
                    # print ' '
        for p in nolink:
            post = du.select(student_set, p[1], '==', 0)
            if not len(post) == 0:
                post = post[0]
                pre = du.select(du.select(student_set, p[0], '==', 0), post[1], '<', 1)
                rem = du.select(du.select(student_set, p[0], '==', 0), post[1], '>', 1)

                if not (len(pre) == 0 or len(rem) == 0):
                    pre = pre[0]
                    rem = rem[0]

                    samp_pre = []
                    samp_post = []
                    samp_rem = []
                    for j in range(3, 8):
                        samp_pre.append(pre[j])
                        samp_post.append(post[j])
                        samp_rem.append(rem[j])

                    samples.append([samp_pre, samp_post, samp_rem, [p[0], p[1]]])
                    labels.append([0, 1, 0])

                    # print pre
                    # print post
                    # print rem
                    # print ' '

# =================================================================

    if len(labels) == 0:
        print "\nNO USABLE SAMPLES EXIST"
        exit()

    du.print_label_distribution(labels, ['Prerequisite','Non-Link','Reversed'])
    samples,labels = du.shuffle(samples, labels)
    return samples,labels

def split_for_autoencoding(samples):
    ini_input = []
    ini_output = []
    rem_input = []
    rem_output = []

    for samp in samples:
        ini_input.append(samp[0])
        ini_output.append(samp[1])
        rem_input.append(samp[1])
        rem_output.append(samp[2])

    return du.convert_to_floats(ini_input), du.convert_to_floats(ini_output),\
           du.convert_to_floats(rem_input), du.convert_to_floats(rem_output)


if __name__ == "__main__":

    # load training and test data

    training = []
    tr_label = []
    testing = []
    test_label = []

    samples,labels = load_skill_data('simulated_data.csv','simulated_hierarchy.csv','simulated_hierarchy_nonlink.csv')
    tr_samples, t_samples, tr_labels,t_labels = du.split_training_test(samples,labels)

    t_tr_labels = du.transpose(tr_labels)
    import math
    pre_rep = int(math.floor((len(t_tr_labels[0]) / np.nansum(t_tr_labels[0])) + 1))
    non_rep = int(math.floor((len(t_tr_labels[1]) / np.nansum(t_tr_labels[1])) + 1))
    rev_rep = int(math.floor((len(t_tr_labels[2]) / np.nansum(t_tr_labels[2])) + 1))

    print pre_rep, non_rep, rev_rep

    re_tr_samples, re_tr_labels = add_representation(tr_samples, tr_labels, 0, pre_rep)
    re_tr_samples, re_tr_labels = add_representation(re_tr_samples, re_tr_labels, 1, non_rep)
    re_tr_samples, re_tr_labels = add_representation(re_tr_samples, re_tr_labels, 2, rev_rep)

    re_tr_samples, re_tr_labels = du.sample(re_tr_samples,re_tr_labels,p=0.2)

    du.print_label_distribution(re_tr_labels, ['Prerequisite', 'Non-Link', 'Reversed'])

    ini_input, ini_output, rem_input, rem_output = split_for_autoencoding(re_tr_samples)

    encoder_hidden = 4
    encoder_step = 0.001
    encoder_drop = 0.2
    encoder_batch = 3
    encoder_epochs = 5

    hidden = 5
    step = 0.001
    drop = 0.2
    batch = 3
    epoch = 10
    folds = 10

    ### TRAINING ###
    AB = AutoEncoder(encoder_hidden)
    AB.set_hyperparams(encoder_step,encoder_drop,encoder_batch,encoder_epochs,folds)
    AB.train(ini_input, ini_output)

    BA = AutoEncoder(encoder_hidden)
    BA.set_hyperparams(encoder_step,encoder_drop,encoder_batch,encoder_epochs,folds)
    BA.train(rem_input, rem_output)

    # AA = AutoEncoder(encoder_hidden)
    # AA.set_hyperparams(encoder_step, encoder_drop, encoder_batch, encoder_epochs, folds)
    # AA.train(ini_input, rem_output)

    # ini_input, ini_output, rem_input, rem_output = split_for_autoencoding(re_tr_samples)

    AB_pred = AB.get_hidden_layer(ini_input)
    BA_pred = BA.get_hidden_layer(rem_input)
    # AA_pred = AA.get_hidden_layer(ini_input)

    merged = []
    for p in range(0,len(AB_pred)):
        s = []
        for h in range(0,len(AB_pred[p])):
            s.append(AB_pred[p][h])
            s.append(BA_pred[p][h])
            # s.append(AA_pred[p][h])
        merged.append(s)

    #CAP = AutoEncoder(20)
    #CAP.set_hyperparams(encoder_step,encoder_drop,encoder_batch,encoder_epochs,folds)
    #CAP.train(merged)

    FF = FFNNet(hidden)
    FF.set_hyperparams(step,drop,batch,epoch,folds)
    FF.train(merged,re_tr_labels)

    ### TESTING ###
    ini_input, ini_output, rem_input, rem_output = split_for_autoencoding(t_samples)

    AB_pred = AB.get_hidden_layer(ini_input)
    BA_pred = BA.get_hidden_layer(rem_input)
    # AA_pred = AA.get_hidden_layer(ini_input)

    merged = []
    for p in range(0, len(AB_pred)):
        s = []
        for h in range(0, len(AB_pred[p])):
            s.append(AB_pred[p][h])
            s.append(BA_pred[p][h])
            # s.append(AA_pred[p][h])
        merged.append(s)

    preds = FF.test(merged,t_labels,['Prerequisite', 'Non-Link', 'Reversed'])

    table = [["Link","Pred_Prereq","Pred_NonLink","Pred_Reverse","Label_Prereq",
              "Label_NonLink","Label_Reverse"]]
    for i in range(0,len(preds)):
        table.append([t_samples[i][3][0]+t_samples[i][3][1],preds[i][0],preds[i][1],preds[i][2],
               t_labels[i][0],t_labels[i][1],t_labels[i][2]])

    du.writetoCSV(table,"prediction_values")