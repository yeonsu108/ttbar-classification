
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
trainInput = "../SNN_sample/ttbar.h5"

epoch = 2
trainOutput="./convSNN_ttbar_4_epoch"+str(epoch)
try: os.mkdir(trainOutput)
except: pass

import sys
sys.stdout = open(trainOutput+"/output.txt", 'w')

import nengo
import nengo_dl
import nengo_loihi
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from myutils import *
nengo_loihi.set_defaults()

def conv_layer(x, *args, activation=True, **kwargs):
    # create a Conv2D transform with the given arguments
    conv = nengo.Convolution(*args, channels_last=False, **kwargs)

    if activation:
        # add an ensemble to implement the activation function
        layer = nengo.Ensemble(conv.output_shape.size, 1).neurons
    else:
        # no nonlinearity, so we just use a node
        layer = nengo.Node(size_in=conv.output_shape.size)

    # connect up the input object to the new layer
    nengo.Connection(x, layer, transform=conv)

    # print out the shape information for our new layer
    print("LAYER")
    print(conv.input_shape.shape, "->", conv.output_shape.shape)

    return layer, conv

def classification_accuracy(y_true, y_pred):
    return 100 * tf.metrics.sparse_categorical_accuracy(y_true[:, -1], y_pred[:, -1])

# set up training data
data = pd.read_hdf(trainInput)
##make the number of events for each category equal
#pd_tth = data[data['category'] == 0].sample(n=40330)
##pd_tth = data[data['category'] == 0].sample(n=0) #without tth
#pd_ttlf = data[data['category'] == 1].sample(n=40330)
#pd_ttb = data[data['category'] == 2].sample(n=40330)
#pd_ttbb = data[data['category'] == 3].sample(n=40330)
#pd_ttc = data[data['category'] == 4].sample(n=40330)
#pd_ttcc = data[data['category'] == 5].sample(n=40330)
#
## merge data and reset index
#pd_data = pd.concat([pd_tth, pd_ttlf, pd_ttb, pd_ttbb, pd_ttc, pd_ttcc], ignore_index=True)
#data = pd_data.sample(frac=1).reset_index(drop=True)

# make the number of events for each category equal
pd_tth = data[data['category'] == 0].sample(n=70000)
pd_ttlf = data[data['category'] == 1].sample(n=70000)
pd_ttb = data[data['category'] == 2].sample(n=35000)
pd_ttbb = data[data['category'] == 3].sample(n=35000)
pd_ttc = data[data['category'] == 4].sample(n=35000)
pd_ttcc = data[data['category'] == 5].sample(n=35000)

pd_ttbb = pd.concat([pd_ttb, pd_ttbb])
pd_ttcc = pd.concat([pd_ttc, pd_ttcc])

pd_ttbb['category'] = 2
pd_ttcc['category'] = 3

# merge data and reset index
pd_data = pd.concat([pd_tth, pd_ttlf, pd_ttbb, pd_ttcc], ignore_index=True)
pd_data = pd_data.sample(frac=1).reset_index(drop=True)


variables = ['ngoodjets', 'nbjets_m', 'nbjets_t', 'ncjets_l',
        'ncjets_m', 'ncjets_t', 'deltaR_j12', 'deltaR_j34',
        'sortedjet_pt1', 'sortedjet_pt2', 'sortedjet_pt3', 'sortedjet_pt4',
        'sortedjet_eta1', 'sortedjet_eta2', 'sortedjet_eta3', 'sortedjet_eta4',
        'sortedjet_phi1', 'sortedjet_phi2', 'sortedjet_phi3', 'sortedjet_phi4',
        'sortedjet_mass1', 'sortedjet_mass2', 'sortedjet_mass3', 'sortedjet_mass4',
        'sortedjet_btag1', 'sortedjet_btag2', 'sortedjet_btag3', 'sortedjet_btag4',
#        'sortedbjet_pt1', 'sortedbjet_pt2', 'sortedbjet_pt3', 'sortedbjet_pt4',
#        'sortedbjet_eta1', 'sortedbjet_eta2', 'sortedbjet_eta3', 'sortedbjet_eta4',
#        'sortedbjet_phi1', 'sortedbjet_phi2', 'sortedbjet_phi3', 'sortedbjet_phi4',
#        'sortedbjet_mass1', 'sortedbjet_mass2', 'sortedbjet_mass3', 'sortedbjet_mass4',
]
class_names = ["tth", "ttlf", "ttb", "ttbb", "ttc", "ttcc"]
class_names = ["tth", "ttlf", "ttbb", "ttcc"]

nVariables, nClass = len(variables), len(class_names)

train_data = np.array(data.filter(items = variables))
train_out  = np.array(data.filter(items = ['category']))

numbertr=len(train_out)
trainlen = int(numbertr * 0.8)

train_out = train_out.reshape( (numbertr, 1) )

valid_data=train_data[trainlen:,0::]
valid_data_out=train_out[trainlen:]

train_data=train_data[:trainlen,0::]
train_data_out=train_out[:trainlen]

dt = 0.001  # simulation timestep
presentation_time = 0.1  # input presentation time
max_rate = 100  # neuron firing rates
# neuron spike amplitude (scaled so that the overall output is ~1)
amp = 1 / max_rate
# input image shape
input_shape = (1, 4, 7)
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
    #neuron_type = nengo_loihi.neurons.LoihiLIF(tau_rc=0.02, tau_ref=0.001, amplitude=amp)
    net.config[nengo.Ensemble].neuron_type = neuron_type

    # the input node that will be used to feed in input images
    inp = nengo.Node(
        nengo.processes.PresentInput(valid_data, presentation_time), size_out=nVariables
    )

    # the output node provides the 10-dimensional classification
    out = nengo.Node(size_in=nClass)

    # build parallel copies of the network
    for _ in range(n_parallel):
        layer, conv = conv_layer(inp, 1, input_shape, kernel_size=(1, 1), init=np.ones((1, 1, 1, 1)))
        # first layer is off-chip to translate the images into spikes
        net.config[layer.ensemble].on_chip = False
        #layer, conv = conv_layer(layer, 2, conv.output_shape, strides=(1, 1))
        layer, conv = conv_layer(layer, nClass, conv.output_shape, strides=(1, 1))

        nengo.Connection(layer, out, transform=nengo_dl.dists.Glorot())

    out_p = nengo.Probe(out, label="out_p")
    out_p_filt = nengo.Probe(out, synapse=nengo.Alpha(0.01), label="out_p_filt")

#set train data
train_data = train_data.reshape(-1, 1, nVariables)
valid_data = valid_data.reshape(-1, 1, nVariables)
train_data_out = train_data_out.reshape(-1, 1, 1)
valid_data_out = valid_data_out.reshape(-1, 1, 1)

train_data = {inp: train_data, out_p: train_data_out}

# for the test data evaluation we'll be running the network over time
# using spiking neurons, so we need to repeat the input/target data
# for a number of timesteps (based on the presentation_time)
test_data = {
    inp: np.tile(valid_data, (1, int(presentation_time / dt), 1)),
    out_p_filt: np.tile(valid_data_out, (1, int(presentation_time / dt), 1))
}

do_training = True
with nengo_dl.Simulator(net, minibatch_size=minibatch_size, seed=0, progress_bar=False) as sim:
    if do_training:
        sim.compile(loss={out_p_filt: classification_accuracy})
        print("accuracy before training: %.2f%%" %
                sim.evaluate(test_data[inp], {out_p_filt: test_data[out_p_filt]}, verbose=0)["loss"])

        # run training
        sim.compile(
                optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
                loss={out_p: tf.losses.SparseCategoricalCrossentropy(from_logits=True)}
        )
        sim.fit(train_data[inp], train_data[out_p], epochs=epoch)

        sim.compile(loss={out_p_filt: classification_accuracy})
        print("accuracy after training: %.2f%%" %
                sim.evaluate(test_data[inp], {out_p_filt: test_data[out_p_filt]}, verbose=0)["loss"])

        sim.save_params(trainOutput+"/model")
    else:
        sim.compile(loss={out_p_filt: classification_accuracy})
        print("accuracy before training: %.2f%%" %
                sim.evaluate(test_data[inp], {out_p_filt: test_data[out_p_filt]}, verbose=0)["loss"])
        sim.load_params(trainOutput+"/model")
        print("parameters loaded")
        sim.compile(loss={out_p_filt: classification_accuracy})
        print("accuracy after training: %.2f%%" %
                sim.evaluate(test_data[inp], {out_p_filt: test_data[out_p_filt]}, verbose=0)["loss"])

    # store trained parameters back into the network
    sim.freeze_params(net)

for conn in net.all_connections:
    conn.synapse = 0.005

if do_training:
    with nengo_dl.Simulator(net, minibatch_size=minibatch_size, progress_bar=False) as sim:
        sim.compile(loss={out_p_filt: classification_accuracy})
        print("accuracy w/ synapse: %.2f%%" %
                sim.evaluate(test_data[inp], {out_p_filt: test_data[out_p_filt]}, verbose=0)["loss"])

n_presentations = 2000
with nengo_loihi.Simulator(net, dt=dt, precompute=False, progress_bar=False) as sim:
    # if running on Loihi, increase the max input spikes per step
    if 'loihi' in sim.sims:
        sim.sims['loihi'].snip_max_spikes_per_step = 120
    #class_names = ["ttlf", "ttb", "ttbb", "ttc", "ttcc"]

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
    comp = sim.data[out_p][step - 1::step]
    _acc = 100 * np.mean(
        np.argmax(comp, axis=-1) == test_data[out_p_filt][:n_presentations, -1, 0]
    )
    print ("comp acc: ", _acc)

    predicted = np.argmax(output, axis=-1)
    correct = test_data[out_p_filt][:n_presentations, -1, 0]

    predicted = np.array(predicted, dtype=int)
    correct = np.array(correct, dtype=int)

    print("Predicted labels: ", predicted[:50])
    print("Correct labels: ", correct[:50])
    print("loihi error: %.2f%%" % error_percentage)
    print("loihi acc: %.2f%%" % acc)

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(correct, predicted, classes=class_names,
                    title='Confusion matrix, without normalization, acc=%.2f'%acc, savename=trainOutput+"/confusion_matrix.pdf")

    # Plot normalized confusion matrix
    plot_confusion_matrix(correct, predicted, classes=class_names, normalize=True,
                    title='Normalized confusion matrix, acc=%.2f'%acc, savename=trainOutput+"/norm_confusion_matrix.pdf")

for i in range(5):
    n_plots = 5
    correct = test_data[out_p_filt][n_plots*i:n_plots*(i+1), -1, 0]
    correct_str = ""
    for j in correct:
        correct_str += class_names[j]+"  "
    correct = "".join(map(str, correct))


    plt.figure()

    plt.plot(sim.trange()[n_plots*i*step:n_plots*(i+1)*step], sim.data[out_p_filt][n_plots*i*step:n_plots*(i+1)*step])
    plt.legend(class_names, loc="best")
    plt.title(correct_str)

    plt.savefig(trainOutput+"/label_"+correct+".pdf")
