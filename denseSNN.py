""" Tagging images of Jets with Convolutional Spiking Neural Networks deployed on the Loihi chip. """

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import nengo
import nengo_dl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import nengo_loihi
from keras.utils.np_utils import to_categorical
from plot_confusion_matrix import plot_confusion_matrix

tf.config.list_physical_devices('GPU')

def classification_accuracy(y_true, y_pred):
    return 100 * tf.metrics.sparse_categorical_accuracy(y_true[:, -1], y_pred[:, -1])

# set up training data
trainInput = "../SNN_sample/ttbar_selected.h5"
epoch = 3
trainOutput="./ttbar_Dense_epoch" + str(epoch)
try: os.mkdir(trainOutput)
except: pass
data = pd.read_hdf(trainInput)
data = data.sample(frac=1).reset_index(drop=True)

variables = ['ngoodjets', 'nbjets_m', 'nbjets_t', 'ncjets_l',
        'ncjets_m', 'ncjets_t', 'sortedjet_pt1', 'sortedjet_pt2',
        'sortedjet_pt3', 'sortedjet_pt4', 'sortedjet_eta1',
        'sortedjet_eta2', 'sortedjet_eta3', 'sortedjet_eta4',
        'sortedjet_phi1', 'sortedjet_phi2', 'sortedjet_phi3',
        'sortedjet_phi4', 'sortedjet_mass1', 'sortedjet_mass2',
        'sortedjet_mass3', 'sortedjet_mass4', 'sortedjet_btag1',
        'sortedjet_btag2', 'sortedjet_btag3', 'sortedjet_btag4',
        'sortedbjet_pt1', 'sortedbjet_pt2', 'sortedbjet_pt3',
        'sortedbjet_pt4', 'sortedbjet_eta1', 'sortedbjet_eta2',
        'sortedbjet_eta3', 'sortedbjet_eta4', 'sortedbjet_phi1',
        'sortedbjet_phi2', 'sortedbjet_phi3', 'sortedbjet_phi4',
        'sortedbjet_mass1', 'sortedbjet_mass2', 'sortedbjet_mass3',
        'sortedbjet_mass4', 'sortedbjet_btag1', 'sortedbjet_btag2',
        'sortedbjet_btag3', 'sortedbjet_btag4', 'deltaR_j12', 'deltaR_j34', 'deltaR_b12']
class_names = np.array(['tthbb', 'ttlf', 'ttb', 'ttbb', 'ttc', 'ttcc'], dtype=str)
nVariables, nClass = len(variables), len(class_names)
print ("nVariables and nClass", nVariables, nClass)
train_data = np.array(data.filter(items = variables))
train_out  = np.array(data.filter(items = ['category']))

numbertr=len(train_out)
train_out = train_out.reshape( (numbertr, 1) )

#Splitting between training set and cross-validation set
trainlen = 222000
valid_data=train_data[trainlen:,0::]
valid_data_out=train_out[trainlen:]

train_data=train_data[:trainlen,0::]
train_data_out=train_out[:trainlen]
train_data_out_categorical=to_categorical(train_data_out)

dt = 0.001  # simulation timestep
presentation_time = 0.1  # input presentation time
max_rate = 100  # neuron firing rates
# neuron spike amplitude (scaled so that the overall output is ~1)
amp = 1 / max_rate
# input image shape
input_shape = (1, 7, 7)
n_parallel = 2  # number of parallel network repetitions
minibatch_size = 200

with nengo.Network(seed=0) as net:
    # set up the default parameters for ensembles/connections
    nengo_loihi.add_params(net)
    net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([max_rate])
    net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
    net.config[nengo.Connection].synapse = None

    #neuron_type = nengo.SpikingRectifiedLinear(amplitude=amp)
    neuron_type = nengo.LIF(tau_rc=0.02, tau_ref=0.001, amplitude=amp)
    #neuron_type = nengo.AdaptiveLIF(amplitude=amp)
    #neuron_type = nengo.Izhikevich()
    # the input node that will be used to feed in input images
    inp = nengo.Node(
        nengo.processes.PresentInput(valid_data, presentation_time), size_out=nVariables
    )

    # the output node provides the 10-dimensional classification
    out = nengo.Node(size_in=nClass)

    layer_1 = nengo.Ensemble(n_neurons=64, dimensions=1, neuron_type=neuron_type, label="Layer 1")
    net.config[layer_1].on_chip = False
    nengo.Connection(inp, layer_1.neurons, transform=nengo_dl.dists.Glorot())
    p1 = nengo.Probe(layer_1.neurons)

    layer_2 = nengo.Ensemble(n_neurons=32, dimensions=1, neuron_type=neuron_type, label="Layer 2")
    nengo.Connection(layer_1.neurons, layer_2.neurons, transform=nengo_dl.dists.Glorot())
    p2 = nengo.Probe(layer_2.neurons)

    layer_3 = nengo.Ensemble(n_neurons=16, dimensions=1, neuron_type=neuron_type, label="Layer 3")
    nengo.Connection(layer_2.neurons, layer_3.neurons, transform=nengo_dl.dists.Glorot())
    p3 = nengo.Probe(layer_3.neurons)

    nengo.Connection(layer_3.neurons, out, transform=nengo_dl.dists.Glorot())

    out_p = nengo.Probe(out, label="out_p")
    out_p_filt = nengo.Probe(out, synapse=nengo.Alpha(0.01), label="out_p_filt")

#set train data
train_data = train_data.reshape(-1, 1, nVariables)
valid_data = valid_data.reshape(-1, 1, nVariables)
train_data_out = train_data_out.reshape(-1, 1, 1)
train_data_out_categorical = train_data_out_categorical.reshape(-1, 1, 2)
valid_data_out = valid_data_out.reshape(-1, 1, 1)

train_data = {inp: train_data, out_p: train_data_out}
#train_data = {inp: train_data, out_p: train_data_out_categorical}

# for the test data evaluation we'll be running the network over time
# using spiking neurons, so we need to repeat the input/target data
# for a number of timesteps (based on the presentation_time)
test_data = {
    inp: np.tile(valid_data, (1, int(presentation_time / dt), 1)),
    out_p_filt: np.tile(valid_data_out, (1, int(presentation_time / dt), 1))
}

do_training = True
with nengo_dl.Simulator(net, minibatch_size=minibatch_size, seed=0) as sim:
    if do_training:
        sim.compile(loss={out_p_filt: classification_accuracy})
        print("accuracy before training: %.2f%%" %
                sim.evaluate(test_data[inp], {out_p_filt: test_data[out_p_filt]}, verbose=0)["loss"])

        # run training
        sim.compile(
                optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
                loss={out_p: tf.losses.SparseCategoricalCrossentropy(from_logits=True)}
                #loss = {out_p: tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2}
        )
        sim.fit(train_data[inp], {out_p: train_data[out_p]}, epochs=epoch)

        sim.compile(loss={out_p_filt: classification_accuracy})
        print("accuracy after training: %.2f%%" %
                sim.evaluate(test_data[inp], {out_p_filt: test_data[out_p_filt]}, verbose=0)["loss"])

        sim.save_params(trainOutput+"/ttbb_params")
    else:
        sim.compile(loss={out_p_filt: classification_accuracy})
        print("accuracy before training: %.2f%%" %
                sim.evaluate(test_data[inp], {out_p_filt: test_data[out_p_filt]}, verbose=0)["loss"])
        sim.load_params("./model/ttbb_params")
        print("parameters loaded")
        sim.compile(loss={out_p_filt: classification_accuracy})
        print("accuracy after training: %.2f%%" %
                sim.evaluate(test_data[inp], {out_p_filt: test_data[out_p_filt]}, verbose=0)["loss"])

    # store trained parameters back into the network
    sim.freeze_params(net)


for conn in net.all_connections:
    conn.synapse = 0.005

if do_training:
    with nengo_dl.Simulator(net, minibatch_size=minibatch_size) as sim:
        sim.compile(loss={out_p_filt: classification_accuracy})
        print("accuracy w/ synapse: %.2f%%" %
                sim.evaluate(test_data[inp], {out_p_filt: test_data[out_p_filt]}, verbose=0)["loss"])


n_presentations = 2000
with nengo_loihi.Simulator(net, dt=dt, precompute=False) as sim:
    # if running on Loihi, increase the max input spikes per step
    if 'loihi' in sim.sims:
        sim.sims['loihi'].snip_max_spikes_per_step = 120

    # run the simulation on Loihi
    sim.run(n_presentations * presentation_time)

    # check classification error
    step = int(presentation_time / dt)
    output = sim.data[out_p_filt][step - 1::step]
    #print ("output", output)
    error_percentage = 100 * np.mean(
        np.argmax(output, axis=-1) != test_data[out_p_filt][:n_presentations, -1, 0]
    )
    acc = 100 * np.mean(
        np.argmax(output, axis=-1) == test_data[out_p_filt][:n_presentations, -1, 0]
    )

    predicted = np.argmax(output, axis=-1)
    correct = test_data[out_p_filt][:n_presentations, -1, 0]

    predicted = np.array(predicted, dtype=int)
    correct = np.array(correct, dtype=int)

    print("Predicted labels: ", predicted)
    print("Correct labels: ", correct)
    print("loihi error: %.2f%%" % error_percentage)
    print("loihi acc: %.2f%%" % acc)

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(correct, predicted, classes=class_names,
                          title='Confusion matrix, without normalizationi, acc='+str(acc))
    plt.savefig(trainOutput+"/confusion_matrix.pdf")

    # Plot normalized confusion matrix
    plot_confusion_matrix(correct, predicted, classes=class_names, normalize=True,
                          title='Normalized confusion matrix, acc='+str(acc))
    plt.savefig(trainOutput+"/norm_confusion_matrix.pdf")

for i in range(1, 6):
    n_plots = 5
    correct = ''.join(map(str,test_data[out_p_filt][n_plots*i:n_plots*(i+1), -1, 0]))

    plt.figure()

    plt.plot(sim.trange()[n_plots*i*step:n_plots*(i+1)*step], sim.data[out_p_filt][n_plots*i*step:n_plots*(i+1)*step])
    plt.legend(class_names, loc="best")
    plt.title(correct)

    plt.savefig(trainOutput+"/label_"+correct+".pdf")
