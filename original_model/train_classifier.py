import tensorflow as tf
import keras.optimizers
import scipy.io
from keras.models import model_from_json
from keras.losses import KLDivergence
import os

import numpy as np
import keras.backend as K
import classifier

from datetime import datetime

def save_model(model, json_path, weight_path):
    json_string = model.to_json()
    open(json_path, 'w').write(json_string)
    dict = {}
    i = 0
    for layer in model.layers:
        weights = layer.get_weights()
        my_list = np.zeros(len(weights), dtype=np.object)
        my_list[:] = weights
        dict[str(i)] = my_list
        i += 1
    scipy.io.savemat(weight_path, dict)

def load_model(json_path):
    model = model_from_json(open(json_path).read())
    return model

def load_batch_train(normal_path, normal_list, abnormal_path, abnormal_list, mirror_normal_path, mirror_abnormal_path):

    batchsize=60
    n_exp = int(batchsize/2)

    num_normal = len(normal_list)
    num_abnormal = len(abnormal_list)

    abnor_list_idx = np.random.permutation(num_abnormal)
    abnor_list = abnor_list_idx[:n_exp]
    norm_list_idx = np.random.permutation(num_normal)
    norm_list = norm_list_idx[:n_exp]

    abnormal_feats = []
    for video_idx in abnor_list:
        video_path = os.path.join(abnormal_path, abnormal_list[video_idx])
        with open(video_path, "rb") as f:
            feats = np.loadtxt(f)
        abnormal_feats.append(feats)

    normal_feats = []
    for video_idx in norm_list:
        video_path = os.path.join(normal_path, normal_list[video_idx])
        with open(video_path, "rb") as f:
            feats = np.loadtxt(f)
        normal_feats.append(feats)


    all_feats = np.vstack((*abnormal_feats, *normal_feats))
    all_labels = np.zeros(32*batchsize, dtype='uint8')

    all_labels[:32*n_exp] = 1

    abnormal_feats = []
    for video_idx in abnor_list:
        video_path = os.path.join(mirror_abnormal_path, abnormal_list[video_idx])
        with open(video_path, "rb") as f:
            feats = np.loadtxt(f)
        abnormal_feats.append(feats)

    normal_feats = []
    for video_idx in norm_list:
        video_path = os.path.join(mirror_normal_path, normal_list[video_idx])
        with open(video_path, "rb") as f:
            feats = np.loadtxt(f)
        normal_feats.append(feats)


    all_feats_mirror = np.vstack((*abnormal_feats, *normal_feats))
    all_labels_mirror = np.zeros(32*batchsize, dtype='uint8')

    all_labels_mirror[:32*n_exp] = 1

    feats = np.vstack((*all_feats, *all_feats_mirror))
    labels = np.vstack((*all_labels, *all_labels_mirror))

    #print(feats.shape)
    #print(labels.shape)

    return  feats, labels


def custom_objective(y_true1, y_pred1):

    y_true1 = K.reshape(y_true1, [-1])
    y_pred1 = K.reshape(y_pred1, [-1])
    y_true = y_true1[:1920]
    y_true_mirror = y_true1[1920:]
    y_pred = y_pred1[:1920]
    y_pred_mirror = y_pred1[1920:]
    n_seg = 32
    nvid = 60
    n_exp = int(nvid / 2)

    max_scores_list = []
    z_scores_list = []
    temporal_constrains_list = []
    sparsity_constrains_list = []

    for i in range(0, n_exp, 1):

        video_predictions = y_pred[i*n_seg:(i+1)*n_seg]

        max_scores_list.append(K.max(video_predictions))
        temporal_constrains_list.append(
            K.sum(K.pow(video_predictions[1:] - video_predictions[:-1], 2))
        )
        sparsity_constrains_list.append(K.sum(video_predictions))

    for j in range(n_exp, 2*n_exp, 1):

        video_predictions = y_pred[j*n_seg:(j+1)*n_seg]
        max_scores_list.append(K.max(video_predictions))

    max_scores = K.stack(max_scores_list)
    temporal_constrains = K.stack(temporal_constrains_list)
    sparsity_constrains = K.stack(sparsity_constrains_list)

    for ii in range(0, n_exp, 1):
        max_z = K.maximum(1 - max_scores[:n_exp] + max_scores[n_exp+ii], 0)
        z_scores_list.append(K.sum(max_z))

    z_scores = K.stack(z_scores_list)
    z = K.mean(z_scores)

    y_true = K.clip(y_true, K.epsilon(), 1)
    y_true_mirror = K.clip(y_pred, K.epsilon(), 1)
    kl_loss=K.sum(y_true * K.log(y_true / y_true_mirror), axis=-1) + K.sum(y_true_mirror * K.log(y_true_mirror / y_true), axis=-1)

    #print(kl_loss)

    return z  + kl_loss + \
        0.00008*K.sum(temporal_constrains) + \
        0.00008*K.sum(sparsity_constrains)

output_dir = "trained_models/"
normal_dir = "C:/Users/dalab/Downloads/full projects/projects/sultani/c3d-32/Train/c3d-normal"
abnormal_dir = "C:/Users/dalab/Downloads/full projects/projects/sultani/c3d-32/Train/c3d-abnormal"

mirror_normal_dir = "C:/Users/dalab/Downloads/full projects/projects/rtfm/Train/c3d/normal-32"
mirror_abnormal_dir = "C:/Users/dalab/Downloads/full projects/projects/rtfm/Train/c3d/abnormal-32"

normal_list = os.listdir(normal_dir)
normal_list.sort()
abnormal_list = os.listdir(abnormal_dir)
abnormal_list.sort()

weights_path = output_dir + 'weights.mat'

model_path = output_dir + 'model.json'

#Create Full connected Model
model = classifier.classifier_model()

adagrad = keras.optimizers.Adagrad(lr=0.001, epsilon=1e-08)
model.compile(loss=custom_objective, optimizer=adagrad)

if not os.path.exists(output_dir):
       os.makedirs(output_dir)

loss_graph =[]
num_iters = 20000
total_iterations = 0
batchsize=60
time_before = datetime.now()


for it_num in range(num_iters):
    inputs, targets = load_batch_train(
        normal_dir, normal_list, abnormal_dir, abnormal_list, mirror_normal_dir, mirror_abnormal_dir
    )
    batch_loss = model.train_on_batch(inputs, targets)
    loss_graph = np.hstack((loss_graph, batch_loss))
    total_iterations += 1
    if total_iterations % 20 == 0:
        print ("Iteration={} took: {}, loss: {}".format(
            total_iterations, datetime.now() - time_before, batch_loss)
        )

print("Train Successful - Model saved")
save_model(model, model_path, weights_path)
