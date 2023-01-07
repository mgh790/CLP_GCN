from __future__ import division
from __future__ import print_function
from scipy import sparse
from scipy import spatial
import pickle
import time
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import scipy
# from  PPMI import *
from utils import *
from models import GCN, MLP
import numpy as np
from sklearn.metrics import f1_score
from scipy import spatial
from collections import Counter
import math
# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_string('model', 'gcn', 'Model string.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs1', 201, 'Number of epochs to train.')
flags.DEFINE_integer('epochs2', 201, 'Number of epochs to train.')
flags.DEFINE_integer('epochs', 20, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.6, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_string('a_plus', 'Salton', '.')

adj, adjW, features, y_train, y_val, y_test, y_unlbl, train_mask, \
val_mask, test_mask, unlbl_mask, ally_lbl, idx_train, idx_val, idx_test = load_data(
    FLAGS.dataset,aplus=FLAGS.a_plus)
num_nodes = adj.shape[0]
features_array = features.toarray()
# Some preprocessing
features = preprocess_features(features)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    support2 = [preprocess_adj(adjW)]
    num_supports = 1
    model_func = GCN

# Define placeholders
num_nodes = adj.shape[0]
num_class = y_train.shape[1]
local_list = np.zeros(shape=(num_nodes, y_train.shape[1]))
weights = np.zeros(shape=[adj.shape[0]])

placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'hiddenOrg': tf.placeholder(tf.float32, shape=(None, FLAGS.hidden1)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
    'localLoss': tf.placeholder_with_default(np.zeros(shape=(num_nodes, y_train.shape[1])),
                                             [num_nodes, y_train.shape[1]]),
    'weights': tf.placeholder_with_default(np.zeros(shape=(num_nodes)), shape=(num_nodes))
}
# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
sess = tf.Session()

# Init variables
sess.run(tf.global_variables_initializer())
if FLAGS.dataset == 'cora':
    degree = pickle.load(open('pickle/degree_cora.pickle', 'rb'))
weights = [degree[index] / max(degree) for index in range(adj.shape[0])]
# Define model evaluation function
def evaluate(features, support,  labels, mask, placeholders, local_list, weights):
    feed_dict_val = construct_feed_dict(features, support,  labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1]
def similarity(adjW):
    sim = np.zeros(shape=(adj.shape[0], adj.shape[0]))
    a = np.zeros(shape=(adj.shape[0], adj.shape[0]))
    degree_sim = np.zeros(shape=(adj.shape[0]))
    fe = features_array
    index = -1
    preidct_lbl = np.zeros(shape=(adj.shape[0]))
    preidct_sim = np.zeros(shape=(adj.shape[0]))
    for row in adjW:
            Dic_lbl = {}
            Dic_sim = {}
            index += 1
            sum = 0
            total_lbl=0
            other_sim=0
            for neighborIdx in range(0, len(row.data)):
                res=spatial.distance.cosine(fe[index], fe[row.indices[neighborIdx]])
                if math.isnan(res)!=True:
                    sum += res
            for neighborIdx in range(0, len(row.data)):
                neighbor = row.indices[neighborIdx]
                if len(row.data) >1:
                    sim[index, neighbor] = spatial.distance.cosine(fe[index], fe[neighbor]) / sum
                else:
                    if math.isnan(spatial.distance.cosine(fe[index], fe[neighbor])) != True:
                        sim[index, neighbor] = spatial.distance.cosine(fe[index], fe[neighbor])
                    else:
                        sim[index, neighbor] = 0
                if train_mask[neighbor] == True:
                    total_lbl += 1
                    lbl = np.argmax(y_train[neighbor])
                    if lbl in Dic_lbl.keys():
                        Dic_lbl[lbl] += 1
                        Dic_sim[lbl] +=sim[index, neighbor]
                    else:
                        Dic_lbl[lbl] = 1
                        Dic_sim[lbl] =sim[index, neighbor]
                else:
                    other_sim+=sim[index, neighbor]
                other_sim/=len(row.data)
                if sim[index, neighbor] > 0.02 + 1 / degree[index]:
                    a[index, neighbor] = 1 / degree[index]
            Dic_normalize = {k: v /total_lbl for k, v in Dic_lbl.items()}
            max_value=1
            for k, v in Dic_normalize.items():
                if v>max_value:
                    max_value=v
                    preidct_lbl[index] = k
                else:
                    preidct_lbl[index] = k
                    max_value=v
            max_value=1
            for k, v in Dic_sim.items():
                if v>max_value:
                    max_value=v
                    preidct_sim[index] = v
                else:
                    preidct_sim[index] = v
                    max_value=v
            if bool(Dic_normalize)==False:
                preidct_lbl[index]=1000
                preidct_sim[index]=0.01*other_sim
            degree_sim[index] = np.sum(a[index, :])
    return degree_sim,preidct_lbl,preidct_sim
degree_sim,predict_lbl_pro,predict_sim_pro = similarity(adjW)
cost_val = []
def pseduo(model, feed_dict):
    # # --------------------------------------------------------------------------------------
    # # --------------------------| f(xi) - y hat | for unlbl data----------------------------
    # # --------------------------y(hat)=argMax(f(xi) ----------------------------------------
    Z = sess.run(model.outputs, feed_dict=feed_dict)
    outPS_max = np.argmax(Z,axis=1)
    psedu = np.zeros(shape=(adj.shape[0], y_train.shape[1]))
    for i in range(adj.shape[0]):
            row = Z[i]  # predict lbl
            for j in range(len(row)):
                if j == outPS_max[i]:
                    if FLAGS.dataset == 'cora' :
                        psedu[i, j] = 1
                    if FLAGS.dataset == 'citeseer'or FLAGS.dataset == 'pubmed':
                        psedu[i, j] = row[j]
                    break
    return list(psedu)
def cal_Weights(model,feed_dict_pseduo,adjW):
    z_normalize = np.zeros(shape=(adj.shape[0], y_train.shape[1]))
    Z = sess.run(model.outputs, feed_dict=feed_dict_pseduo)
    outPS_max = np.argmax(Z,axis=1)
    for i in range(adj.shape[0]):
        sigma = np.sum(Z[i, :])
        z_normalize[i, :] =np.divide(Z[i, :], sigma)
        pr_lblpro = int(predict_lbl_pro[i])
        predict_lbl_nn = outPS_max[i]
        if FLAGS.dataset == 'cora':
            if int(pr_lblpro) == int(predict_lbl_nn):
                weights[i]=1
            else:
                weights[i] = np.mean(z_normalize[i, :]) + degree_sim[i]
    return weights
val_array=[]
for epoch in range(FLAGS.epochs2):
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support,  y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # --------------------------------------------------------------------------------------
    # ---------------------------Training step----------------------------------------------
    # --------------------------------------------------------------------------------------
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
    # # ------------------------------------------------------------------------------------
    # # ---------------------------pseduo step----------------------------------------------
    # # ------------------------------------------------------------------------------------
    feed_dict_pseduo = construct_feed_dict(features, support,  y_train, unlbl_mask, placeholders)
    local_list = pseduo(model, feed_dict_pseduo)
    feed_dict_pseduo.update({placeholders['localLoss']: local_list})
    weights = cal_Weights(model,feed_dict_pseduo,adjW)
    feed_dict_pseduo.update({placeholders['weights']: weights})
    sess.run([model.opt_opPS, model.lossPS], feed_dict=feed_dict_pseduo)
    # --------------------------------------------------------------------------------
    # ----------------------------Validation------------------------------------------
    # --------------------------------------------------------------------------------
    cost, acc= evaluate(features, support,  y_val, val_mask, placeholders,
                                                       local_list, weights)
    cost_val.append(cost)
    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc)
          )

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
        print("Early stopping...")
        break
    test_cost, test_acc= evaluate(features, support,  y_test,
                                                                      test_mask, placeholders, local_list,
                                                                      weights)
    val_array.append(test_acc*100)

    if epoch % 20 == 0:
        # Testing
        test_cost, test_acc= evaluate(features, support,  y_test,
                                                                          test_mask, placeholders, local_list,
                                                                          weights)
        print('##############################################################')
        print("Epoch:", '%04d' % (epoch + 1),"Test set results:", "cost=", "{:.5f}".format(test_cost),
              "accuracy=", "{:.5f}".format(test_acc)
              )
        print('##############################################################')

print("Optimization Finished!")

# # Testing
test_cost, test_acc= evaluate(features, support,  y_test, test_mask,
                                                                  placeholders, local_list,
                                                                  weights)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc))
