import os
import time
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import arch
import dl_utils


def graph_builder_wrapper_PNN(input_dict, build_func=arch.mlp, lr_initial=0.01, max_save=100, feed_u=False):
    """Same as graph_builder_wrapper except constrains gradients to be along initial lines
       CURRENTLY ONLY WORKS WITH mlp_noreluout_lastlayernottrainable
    """

    graph = build_func(input_dict)
    
    # Loss
    y = tf.placeholder(tf.float32, shape=[None,1])
    total_loss = dl_utils.loss(y, graph['yhat'])
    
    # W gap
    w = {k:tf.placeholder(tf.float32, shape=[None,None]) for k in graph if 'weights' in k}
    total_w_gap = dl_utils.w_gap(w, {k:graph[k] for k in graph if 'weights' in k})
    
    # Find unit vectors (feed_u recommended since due to numerical issues, the line may drift)
    if feed_u:
        u = tf.Variable(np.zeros([input_dict['p'], input_dict['h']]).astype('float32'), trainable=False)
    else:
        u = graph['weights1'] / tf.sqrt(tf.reduce_sum(tf.square(graph['weights1']), 0, keep_dims=True))
    
    # Optimizer
    learning_rate = tf.Variable(lr_initial, name='learning_rate', trainable=False)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    grads = optimizer.compute_gradients(total_loss)
    modified_grads = []
    for gv in grads:
        if 'w1' in gv[1].name: 
            # Project gradients along unit vectors
            alpha = tf.diag_part(tf.matmul(tf.transpose(u),gv[0]))
            update = u*alpha
            modified_grads.append((update, gv[1]))
        else: 
            modified_grads.append(gv)
            
    opt_step = optimizer.apply_gradients(modified_grads)
    
    graph['y'] = y
    graph['w'] = w
    graph['u'] = u
    graph['grads'] = [i for i in grads if i[0] is not None]
    graph['modified_grads'] = [i for i in modified_grads if i[0] is not None]
    graph['opt_step'] = opt_step
    graph['total_loss'] = total_loss
    graph['total_w_gap'] = total_w_gap
    graph['learning_rate'] = learning_rate
    graph['saver'] = tf.train.Saver(max_to_keep=100)
    return graph


def check_to_see_if_col_are_same_line(X1,X2,verbose=True):
    """Used to check to make sure final weights are aligned with initial weights"""

    u1 = X1 / np.linalg.norm(X1,axis=0)
    u2 = X2 / np.linalg.norm(X2,axis=0)
    dot_products = np.abs(np.diag(np.dot(u1.T,u2)))
    if verbose:
        # for i in dot_products: print(i)
        if np.all(np.isclose(dot_products,1)):
            print('columns along same line')
        else:
            print('columns not along same line')            
    return dot_products


def get_acc(X, Y, w, input_dict, build_func):
    """Estimates labels predicted by network and outputs accuracy for -1, 1 labels"""

    Yhat = dl_utils.generate_output(X, w, input_dict, build_func=build_func)
    return np.sum(np.sign(Yhat) == Y)/float(len(Y))


def get_acc_only_1_2_labels(X, Y, w, input_dict, build_func):
    """Estimates labels predicted by network and outputs accuracy for 1, 2 labels"""

    Yhat = dl_utils.generate_output(X, w, input_dict, build_func=build_func)
    Yhat_out = np.zeros(len(Yhat))
    for i, y in enumerate(Yhat):
        Yhat_out[i] = np.argmin(np.abs(y-np.array([1, 2])))+1
    return np.sum(Yhat_out == Y.reshape(-1))/float(len(Y))


def build_graph_and_train(Xtr, Ytr, Xtt, Ytt, input_dict, build_func, save_file,
                          num_epochs=100, batch_size=100, w_initial=None, PNN=False):
    """builds graph and trains network, saving weights"""

    if not os.path.exists(save_file):
        tf.reset_default_graph()
        if not PNN:
            g = dl_utils.graph_builder_wrapper(input_dict, build_func=build_func)
            out = dl_utils.train_no_wtrue(Xtr, Ytr, g, num_epochs, batch_size,
                                          w_initial=w_initial, normalize_loss=True)
        else:
            g = graph_builder_wrapper_PNN(input_dict, build_func=build_func, feed_u=True)
            out = dl_utils.train_no_wtrue(Xtr, Ytr, g, num_epochs, batch_size,
                                          w_initial=w_initial, PNN=True, normalize_loss=True)

        pickle.dump(out, file(save_file, 'wb'))
    else:
        out = pickle.load(file(save_file, 'rb'))
        
    if PNN:
        dot_prods = check_to_see_if_col_are_same_line(out[1]['weights1'], w_initial['weights1'], verbose=False)
        print('Num final lines close to original: %s' \
              %(np.sum(np.isclose(dot_prods, np.ones(len(dot_prods)), atol=1e-2))))

    tr_acc = get_acc_only_1_2_labels(Xtr, Ytr, out[1], input_dict, build_func)
    test_acc = get_acc_only_1_2_labels(Xtt, Ytt, out[1], input_dict, build_func)

    print('Train acc: %.5f'%(tr_acc))
    print('Test acc: %.5f'%(test_acc))
    
    
def train_set_of_PNNs(Xtr, Ytr, Xtt, Ytt, k_list, d, build_func, save_pref,
                      num_epochs=100, batch_size=100, w2_init_mode=1):
    """Train a set of PNNs based on number of hidden nodes"""
    
    assert w2_init_mode in [1, 2, 3]
    
    # Performance with a PNN network
    for k in k_list:

        input_dict = dict(p=d, h=k)

        # initialize unit vectors corresponding to lines
        np.random.seed(k)
        w_init_u = np.random.normal(0, 1, [d, k/2])
        w_init_u /= np.linalg.norm(w_init_u, axis=0)
        w_init_u = np.repeat(w_init_u, 2, axis=1)
        alpha = np.random.uniform(0, 1, k)
        for ii in range(0, len(alpha), 2): alpha[ii] *= -1
        w_init = alpha*w_init_u
        
        if w2_init_mode == 1:
            w2 = np.ones([k, 1]).astype(np.float32)
        elif w2_init_mode == 2:
            w2 = np.random.normal(0, 1./np.sqrt(k), [k, 1]).astype(np.float32)
        else:
            w2 = np.abs(np.random.normal(0, 1./np.sqrt(k), [k, 1])).astype(np.float32)
            
        w_init = {'weights1':w_init.astype(np.float32),
                  'weights2':w2}

        # start training
        save_file = '%s_hidden%s.pickle'%(save_pref, k)
        build_graph_and_train(Xtr, Ytr, Xtt, Ytt, input_dict, build_func, save_file,
                              num_epochs=num_epochs, batch_size=batch_size, w_initial=w_init, PNN=True)
        
        
def plot_losses(k_list, save_pref):
    """plots losses from saved weights"""

    for k in k_list:
        save_file = '%s_hidden%s.pickle'%(save_pref, k)
        loss = pickle.load(file(save_file, 'rb'))[0]
        plt.plot(loss, label='PNN, %s hidden'%(k))