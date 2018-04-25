from __future__ import print_function
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import arch,time,os

# ------------------------------------------------------------------------------
# Building the network

def loss(y,yhat):
    return tf.reduce_mean(tf.squared_difference(y,yhat), name='mse')

def w_gap(w,what):
    return tf.reduce_sum([tf.reduce_sum(tf.squared_difference(w[k],what[k])) for k in w])

def graph_builder_wrapper(input_dict,build_func=arch.mlp,lr_initial=0.01,max_save=100):
    y = tf.placeholder(tf.float32, shape=[None,1])
    graph = build_func(input_dict)

    w = {k:tf.placeholder(tf.float32, shape=[None,None]) for k in graph if 'weights' in k}

    total_loss = loss(y, graph['yhat'])
    total_w_gap = w_gap(w, {k:graph[k] for k in graph if 'weights' in k})
    learning_rate = tf.Variable(lr_initial, name='learning_rate', trainable=False)
    opt_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)
    graph['y'] = y
    graph['w'] = w
    graph['opt_step'] = opt_step
    graph['total_loss'] = total_loss
    graph['total_w_gap'] = total_w_gap
    graph['learning_rate'] = learning_rate
    graph['saver'] = tf.train.Saver(max_to_keep=100)
    return graph


# ------------------------------------------------------------------------------
# Training the network

def compute_wgap_and_loss(X,Y,w_true,graph,sess,batch_size=100):
    l = []
    for i in range(0,len(X),batch_size):
        x,y = X[i:i+batch_size], Y[i:i+batch_size]
        l.append(sess.run(graph['total_loss'],feed_dict={graph['x']:x,graph['y']:y}))
    wg = sess.run(graph['total_w_gap'],feed_dict={graph['w'][k]:w_true[k] for k in w_true})
    return wg, np.mean(l)

# w_initial should be a dictionary with keys corresponding to weight matrices in the graph
def train(X,Y,graph,num_epochs,batch_size,w_true,w_initial=None,verbose=True,savedir=None,
          early_stop_loss=1e-5,lr_initial=0.01,get_update_history=False, seed=0):

    if verbose: start = time.time()
    training_losses = []
    training_w_gaps = []
    if get_update_history: weights_history = []
    tf.set_random_seed(seed)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_prop))) as sess:
        sess.run(tf.global_variables_initializer())
        if w_initial is not None: 
            for k in w_initial: sess.run(graph[k].assign(w_initial[k]))
                
        # Compute initial w gap and loss
        initial_train_w_gap,initial_train_loss = compute_wgap_and_loss(X,Y,w_true,graph,sess,
                                                                       batch_size=batch_size)
        
        for epoch in range(num_epochs):
            lr = lr_initial*0.95**(epoch/390.) # initial lr * decay rate ^(step/decay_steps)
            sess.run(graph['learning_rate'].assign(lr))
            t = time.time()
            training_loss = 0
            training_w_gap = 0
            steps = 0.
            X_, Y_ = shuffle(X,Y)
            
            for i in range(0,len(X),batch_size):
                x,y = X_[i:i+batch_size], Y_[i:i+batch_size]
                feed_dict = {graph['w'][k]:w_true[k] for k in w_true}
                feed_dict[graph['x']] = x
                feed_dict[graph['y']] = y
                if get_update_history:
                    training_loss_,training_w_gap_,_,grads,mgrads,w = sess.run([graph['total_loss'],
                                                                                graph['total_w_gap'],
                                                                                graph['opt_step'],
                                                                                graph['grads'],
                                                                                graph['modified_grads'],
                                                                                graph['weights1']],
                                                                               feed_dict=feed_dict)
                    weights_history.append((grads,mgrads,w))

                else:
                    training_loss_,training_w_gap_,_ = sess.run([graph['total_loss'],
                                                                 graph['total_w_gap'],
                                                                 graph['opt_step']],
                                                                feed_dict=feed_dict)
                
                training_loss += training_loss_
                training_w_gap += training_w_gap_
                steps += 1.
                
                if verbose:
                    print('\rEpoch %s/%s (%.3f s), batch %s/%s (%.3f s): loss %.3f, w gap: %.3f'
                          %(epoch+1,num_epochs,time.time()-start,steps,
                            len(X_)/batch_size,time.time()-t,
                            training_loss_,training_w_gap_),end='')
            training_losses.append(training_loss/steps)
            training_w_gaps.append(training_w_gap/steps)

            if early_stop_loss is not None and np.mean(training_losses[-10:]) <= early_stop_loss:
                if verbose:
                    print('\rMean loss <= %.3e for last 10 epochs. Stopping training after epoch %s/%s.'
                          %(early_stop_loss,epoch+1,num_epochs+1),end='')
                break

        w_hat = {k:sess.run(graph[k]) for k in graph if 'weights' in k} # grab all final weights
        final_w_gap,final_train_loss = compute_wgap_and_loss(X,Y,w_true,graph,sess,
                                                             batch_size=batch_size)
    
        # only saving weights from last epoch
        if savedir is not None: 
            os.system('mkdir -p %s'%(savedir))
            graph['saver'].save(sess,'%sepoch%s'%(savedir,epoch))
            
    if verbose: print('')
    if get_update_history:
        return training_losses,training_w_gaps,w_hat, \
               initial_train_loss,initial_train_w_gap,final_train_loss,final_w_gap, \
               weights_history
    return training_losses,training_w_gaps,w_hat, \
           initial_train_loss,initial_train_w_gap,final_train_loss,final_w_gap


# w_initial should be a dictionary with keys corresponding to weight matrices in the graph
def train_no_wtrue(X, Y, graph, num_epochs, batch_size, 
                   w_initial=None, verbose=True,savedir=None, early_stop_loss=1e-5,
                   lr_initial=0.01, get_update_history=False, gpu_prop=0.1, seed=0, PNN=False,
                   normalize_loss=False):
    if PNN:
        assert w_initial is not None
        u = w_initial['weights1'] / np.linalg.norm(w_initial['weights1'], axis=0)

    if verbose: start = time.time()
    training_losses = []
    training_w_gaps = []
    if get_update_history: weights_history = []
    tf.set_random_seed(seed)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_prop))) as sess:
        
        sess.run(tf.global_variables_initializer())
        
        if w_initial is not None: 
            for k in w_initial: sess.run(graph[k].assign(w_initial[k]))
                
        if PNN:
            sess.run(graph['u'].assign(u))
        
        for epoch in range(num_epochs):
            lr = lr_initial*0.95**(epoch/390.) # initial lr * decay rate ^(step/decay_steps)
            sess.run(graph['learning_rate'].assign(lr))
            t = time.time()
            training_loss = 0

            steps = 0.
            X_, Y_ = shuffle(X,Y)
            for i in range(0,len(X),batch_size):
                x, y = X_[i:i+batch_size], Y_[i:i+batch_size]
                
                feed_dict = {graph['x']: x, graph['y']: y}
                    
                if get_update_history:
                    training_loss_, _, grads, mgrads, w = sess.run([graph['total_loss'],
                                                                    graph['opt_step'],
                                                                    graph['grads'],
                                                                    graph['modified_grads'],
                                                                    graph['weights1']],
                                                                   feed_dict=feed_dict)
                    weights_history.append((grads, mgrads, w))

                else:
                    training_loss_, _ = sess.run([graph['total_loss'], graph['opt_step']],
                                                 feed_dict=feed_dict)
                
                if normalize_loss:
                    training_loss_ *= len(y)/np.sum(np.square(y))

                training_loss += training_loss_
                steps += 1.
                
                if verbose:
                    print('\rEpoch %s/%s (%.3f s), batch %s/%s (%.3f s): loss %.3f'
                          %(epoch+1, num_epochs, time.time()-start, steps,
                            len(X_)/batch_size, time.time()-t, training_loss_), end='')

            training_losses.append(training_loss/steps)

            if early_stop_loss is not None and np.mean(training_losses[-10:]) <= early_stop_loss:
                if verbose:
                    print('\rMean loss <= %.3e for last 10 epochs. Stopping training after epoch %s/%s.'
                          %(early_stop_loss,epoch+1,num_epochs+1),end='')
                break

        w_hat = {k:sess.run(graph[k]) for k in graph if 'weights' in k} # grab all final weights
        
        # only saving weights from last epoch
        if savedir is not None: 
            os.system('mkdir -p %s'%(savedir))
            graph['saver'].save(sess,'%sepoch%s'%(savedir,epoch))
            
    if verbose: print('')
    if get_update_history:
        return training_losses, w_hat, weights_history
    
    return training_losses, w_hat


# ------------------------------------------------------------------------------
# Computing useful values from the network

def compute_w_gap(w1, w2):
    return np.sum([np.linalg.norm(w1[k]-w2[k])**2 for k in w1])

# Generate random labels with underlying true weights
def generate_random_weights_and_output(X, input_dict, build_func=arch.mlp, batch_size=100, seed=0, gpu_prop=0.1):    
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    graph = build_func(input_dict)
    y = np.zeros((len(X),1))
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_prop))) as sess:
        sess.run(tf.global_variables_initializer())
        w = {k:sess.run(graph[k]) for k in graph if 'weights' in k} # grab all weights for graph
        for i in range(0,len(X),batch_size):
            y_ = sess.run(graph['yhat'],feed_dict={graph['x']: X[i:i+batch_size]})
            y[i:i+batch_size] = y_
    return y,w

# Generate labels with underlying true weights
def generate_output(X, w, input_dict, build_func=arch.mlp, batch_size=100, gpu_prop=0.1):
    tf.reset_default_graph()
    graph = build_func(input_dict)
    y = np.zeros((len(X),1))
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_prop))) as sess:
        sess.run(tf.global_variables_initializer())
        for k in w: sess.run(graph[k].assign(w[k]))
        for i in range(0,len(X),batch_size):
            y_ = sess.run(graph['yhat'],feed_dict={graph['x']:X[i:i+batch_size]})
            y[i:i+batch_size] = y_
    return y

# Generate loss for some given weights w
def generate_loss(X, Y, w, input_dict, build_func=arch.mlp, batch_size=100, gpu_prop=0.1):
    tf.reset_default_graph()
    graph = graph_builder_wrapper(input_dict,build_func=build_func)
    L = []
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_prop))) as sess:
        sess.run(tf.global_variables_initializer())
        for k in w: sess.run(graph[k].assign(w[k]))
        for i in range(0,len(X),batch_size):
            feed_dict = {graph['x']: X[i:i+batch_size], graph['y']: Y[i:i+batch_size]}
            L.append(sess.run(graph['total_loss'],feed_dict=feed_dict))
    return np.mean(L)

# Compute loss for multiple combinations of w1, w2
def compute_L(X, Y, all_wi, all_wj, input_dict, build_func=arch.mlp_simple):
    n = len(all_wi)
    L = np.zeros((n,n))
    start = time.time()
    for i,wi in enumerate(all_wi):
        for j,wj in enumerate(all_wj):
            w = {'weights':np.array([[wi],[wj]])}
            L[i,j] = generate_loss(X,Y,w,input_dict,build_func=build_func)
            print('\r%s/%s, %s/%s (%.2f s elapsed)'%(i+1,n,j+1,n,time.time()-start),end='')
    return L

# Get hidden node states (either on or off) using given weights
# weights is a dictionary with keys corresponding to weight matrices in the graph
def get_hidden_states(X, input_dict, weights, build_func=arch.mlp, batch_size=100, gpu_prop=0.1):
    tf.reset_default_graph()
    graph = build_func(input_dict)
    states = []
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_prop))) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(0,len(X),batch_size):
            feed_dict = {graph[k]:weights[k] for k in weights}
            feed_dict[graph['x']] = X[i:i+batch_size]
            if 'hid' in graph: hout_ = sess.run(graph['hid'],feed_dict=feed_dict)
            else: hout_ = sess.run(graph['yhat'],feed_dict=feed_dict)
            states.append(hout_)
    return np.vstack(states) > 0


# Get everything the train function would return
def get_train_out(X, Y, w, input_dict, savedir, build_func, gpu_prop=0.1):
    tf.reset_default_graph()
    graph = graph_builder_wrapper(input_dict,build_func=build_func)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_prop))) as sess:
        epoch = os.listdir(savedir)[0].split('.')[0].split('epoch')[1]
        graph['saver'].restore(sess,'%sepoch%s'%(savedir,epoch))
        w_hat = {k:sess.run(graph[k]) for k in graph if 'weights' in k}
        w_gap_,loss_ = compute_wgap_and_loss(X,Y,w,graph,sess)
    return w_hat,loss_,w_gap_


# ------------------------------------------------------------------------------
# Functions for generating data

def generate_X(N,q,input_dict,cov_is_eye=False,seed=0):

    np.random.seed(seed)

    p = input_dict['p']

    if cov_is_eye:
        # generate covariance matrix approach 1: all iid
        C = np.eye(p)
        A = np.eye(p)

    else:
        # generate covariance matrix approach 2: low-rank covariance matrix
        A = np.random.normal(0,1,[p,q])
        C = np.dot(A,A.T)
        u,s,v = np.linalg.svd(C)
        C /= np.sum(s[:q])/p               # Scale C to ensure eigenvalues sum up to p
        u,v = np.linalg.eigh(C)
        u[u < 0] = 0
        A = np.dot(v,np.diag(np.sqrt(u)))
        B = np.random.normal(0,1,(10,10))  # Rotate A by random matrix to ensure samples are dense
        u,v = np.linalg.eigh(C)
        A = np.dot(A,v)

    X = np.dot(np.random.normal(0,1,[N,p]),A.T) # sample X
    return X


def generate_data(N,q,input_dict,seed=0,build_func=arch.mlp,get_hs=False,cov_is_eye=False):
    np.random.seed(seed)
    X = generate_X(N,q,input_dict,cov_is_eye=cov_is_eye,seed=seed)
    Y,weights = generate_random_weights_and_output(X,input_dict,build_func=build_func)
    if get_hs: 
        hs = get_hidden_states(X,input_dict,weights,build_func=build_func)
        return X,Y,hs,weights
    return X,Y,weights


# ------------------------------------------------------------------------------
