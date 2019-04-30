import optparse
import os
import time
import _pickle as cPickle
import numpy as np
from sklearn.model_selection import train_test_split
import networkx as nx
import tensorflow as tf

"""
Experiments
 (1) [graph classification] Patchy-San + 5 Ego-Convolution(no weight-tying)
 (2) [scale-free regularizer] Patchy-San + 5 Ego-Convolution(weight-tying)
 (3) [base model for scale-free regularizer experiment] Patchy-San + 1 Ego-Convolution
 (4) [unused baseline] Patchy-San + 5 Convolution
--------------------------------------------
Layer Settings
    - Patchy-San: embed labeled neighborhood graphs
        [k=10] neighborhood graphs
         * only 1-hop neighbors are selected
         * node labels as one-hot vector
         * edge labels as weighted edge in adjacency matrix
         * node attribute
         * 1-WL to normalize neighbors[higher priority for less frequent 1-WL labels]
    - Ego-Convolution: enlarge neighborhood graphs
        [k=8,16]
        * only 1-hop neighbors or upto k neighbors by BFS
        * 1-WL to normalize neighbors[higher priority for less frequent 1-WL labels]
----------------------------------------------
"""


PROC_DIR = 'proc'
DATASET_DIR = 'dataset'
DATASET_LIST = ['MUTAG', 'PTC_MR', 'PROTEINS', 'NCI1', 'DD', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINAY', 'COLLAB', 'Compound_Alk-Alc', 'Compound_Asym-Sym']

def parse_arg():
    """utility function to parse argument from command line"""
    parser = optparse.OptionParser()
    parser.add_option('-n', dest='dataset', help='specify the name of dataset in one of {}'.format(DATASET_LIST))
    parser.add_option('-g', dest='gpu_id', default='0', help='specify to run on which GPU')
    parser.add_option('-f', dest='gpu_frac', default='0.99', help='specify the memory utilization of GPU')
    parser.add_option('-k', dest='K', help='hyperparameter k(#neighobr) of Ego-Convolution')
    parser.add_option('-p', dest='PK', default='10', help='hyperparameter k(size of neighbrohood of Patchy-San')
    parser.add_option('-l', dest='loop', default='5', help='#rerun in each fold')
    parser.add_option('-L', dest='lr', default='1e-4', help='learning rate for training')
    parser.add_option('-b', dest='bsize', default='32', help='batch size for training')
    parser.add_option('-c', dest='FOLD', default='10', help='#fold in CV')
    parser.add_option('-m', dest='model_type', default='6L', help="specify which model to use in one of ['6L', '2L', '6L_SF', '3L']")
    (options, args) = parser.parse_args()
    return options
def prepare_G(G_adj, G_nbr, G_nn, G_att, V_num, E_map, N_map, k, dim_f, dim_a, pk=10):
    """
    padded each graph with same # of nodes,
    and each node with same # of neighbors
    --------------------------------------
    return
        adjs: 1 x (|V|+1) x (k x k)
        nbrs: 1 x (|V|+1) x k
        nnlbls: 1 x (|V|+1) x (k x dim_f)
    """
    dim_e = len(E_map)
    adjs, nbrs, nnlbls, atts = [], [], [], []
    pad_adj, pad_nn, pad_att, pad_nbr = np.zeros((pk,pk,dim_e)).reshape((-1)), np.zeros((pk,dim_f)).reshape((-1)), np.zeros((pk,dim_a)).reshape((-1)), np.zeros(k)
    for Adj, Nbr in zip(G_adj, G_nbr):
        adj, nbr, nn, att = [pad_adj], [pad_nbr], [pad_nn], [pad_att]
        for i in range(V_num):
            if i in Adj.keys():
                adj.append(onehot(Adj[i].reshape((1,-1)).astype(np.int32), E_map).reshape((-1)))
                nbr.append(np.array([j+1 for j in Nbr[i]]))
                nnl = onehot(G_nn[i], N_map)
                att_i = G_att[i]
                att_i = att_i if att_i is not None else np.zeros((nnl.shape[0], 0))
                pad_n = pk-nnl.shape[0]
                if pad_n > 0:
                    nnl = np.vstack([nnl, np.zeros((pad_n,dim_f))]) if dim_f > 0 else None
                    att_i = np.vstack([att_i, np.zeros((pad_n,dim_a))])
                nn.append(nnl.reshape((-1)) if dim_f > 0 else pad_nn)
                att.append(att_i.reshape((-1)))
            else:
                adj.append(pad_adj)
                nbr.append(pad_nbr)
                nn.append(pad_nn)
                att.append(pad_att)
        adjs.append(adj)
        nbrs.append(nbr)
        nnlbls.append(nn)
        atts.append(att)
    return np.array(adjs), np.array(nbrs).astype(np.int32), np.array(nnlbls), np.array(atts)
def onehot(labels, Y_map):
    """onehot encoding labels"""
    labels = np.array(labels).reshape((-1))
    N, nY = len(labels), len(Y_map)
    rst = np.zeros((N, nY))
    for i in range(N):
        if labels[i] in Y_map.keys():
            rst[i,Y_map[labels[i]]] = 1
    return rst.astype(np.int32)
def batch_generator(adjs, nbrs, nnlbls, atts, labels, N_map, E_map, Y_map, V_num=28, bsize=32, dim_f=0, dim_a=0, dim_e=0, nY=2, k=3, pk=10):
    """graph is processed(add padding) as needed"""
    epch = 0
    N = len(labels)
    while True:
        order = np.random.permutation(N)
        for i in range(0, N-bsize, bsize):
            Xs = [prepare_G(adjs[x], nbrs[x], nnlbls[x], atts[x], V_num, E_map, N_map, k, dim_f, dim_a, pk=pk) for x in order[i:i+bsize]]
            adj, nbr, nnlbl, att, lbls = [[Xs[x][0] for x in range(len(Xs))]], [Xs[x][1] for x in range(len(Xs))], [Xs[x][2] for x in range(len(Xs))], [Xs[x][3] for x in range(len(Xs))], onehot([labels[x] for x in order[i:i+bsize]], Y_map)
            adj = np.swapaxes(adj, 0,1).reshape((1, bsize, V_num+1, pk*pk*dim_e))
            nbr = np.swapaxes(nbr, 0,1).reshape((1, bsize, V_num+1, k))
            nnlbl = np.swapaxes(nnlbl, 0,1).reshape((1, bsize, V_num+1, pk*dim_f))
            att = np.swapaxes(att, 0,1).reshape((1, bsize, V_num+1, pk*dim_a))
            yield [adj, nbr, nnlbl, att, lbls, epch]
        epch += 1
def xavier_init(size):
    return tf.random_normal(shape=size, stddev=1./tf.sqrt(size[0]/2.))
def xavier_init_val(size):
    return np.random.normal(size=size, scale=1./np.sqrt(size[0]/2.))
def bn(X, eps=1e-8, g=None, b=None):
    if X.get_shape().ndims == 4:
        mean = tf.reduce_mean(X, [0,1,2])
        std = tf.reduce_mean( tf.square(X-mean), [0,1,2])
        X = (X-mean) / tf.sqrt(std+eps)
        if g is not None and b is not None:
            g, b = tf.reshape(g, [1,1,1,-1]), tf.reshape(b, [1,1,1,-1])
            X = X*g + b
    elif X.get_shape().ndims == 2:
        mean = tf.reduce_mean(X, 0)
        std = tf.reduce_mean(tf.square(X-mean), 0)
        X = (X-mean) / tf.sqrt(std+eps)
        if g is not None and b is not None:
            g, b = tf.reshape(g, [1,-1]), tf.reshape(b, [1,-1])
            X = X*g + b
    else:
        raise NotImplementedError
    return X
def aggregate_nbr(pb_nbr_dict, pb_fmap, k):
    """
    nbr_dict: [1, bsize, nV, k]
    fmap: [bsize, nV, nf]
    return [1, bsize, nV, k, nf]
    """
    return tf.stack([
            [tf.nn.embedding_lookup(fmap, nbr_dict)\
             for b_nbr_dict, b_fmap in zip(tf.unstack(pb_nbr_dict), tf.unstack(pb_fmap))\
             for nbr_dict, fmap in zip(tf.unstack(b_nbr_dict), tf.unstack(b_fmap))
            ]])
def g_conv_bn(X, theta, shape):
    """
    X: [1, bsize, nV, n1] => [1, bsize*nV, n1] == swap-axis ==> [bsize*nV, n1]
    W: [n1, n2]
    XW: [bsize*nV, 1, n2] == swap-axis => [1, bsize, nV, n2]
    """
    _, bsize, nV, n2 = shape
    X = tf.reshape(tf.transpose(tf.reshape(X, [1, bsize*nV, -1]), [1,0,2]), [bsize*nV, -1])
    XW = tf.reshape(tf.transpose(tf.reshape(tf.nn.relu(bn(tf.matmul(X, theta[0])+theta[1])), [bsize*nV, 1, n2]), [1,0,2]), [1, bsize, nV, n2])
    return XW
def build_model(nV, dim_f, dim_a, dim_e, bsize=32, k=10, nY=2, pk=10, model_type='6L'):
    """switch experiment models"""
    if model_type == '2L': # Patchy-San + 1 Ego-Convolution for scale-free regularizer exp
        return model_2L(nV, dim_f, dim_a, dim_e, bsize=bsize, k=k, nY=nY, pk=pk)
    elif model_type == '6L_SF': # Patchy-San + 5 Ego-Convolution (without weight-tying) for scale-free regularizer exp
        return model_6L_SF(nV, dim_f, dim_a, dim_e, bsize=bsize, k=k, nY=nY, pk=pk)
    elif model_type == '3L': # for visualization
        return model_3L(nV, dim_f, dim_a, dim_e, bsize=bsize, k=k, nY=nY, pk=pk)
    # [default] Patchy-San + 5 Ego-Convolution (without weight-tying) for graph classificaiton exp
    return model_6L(nV, dim_f, dim_a, dim_e, bsize=bsize, k=k, nY=nY, pk=pk)
def model_6L(nV, dim_f, dim_a, dim_e, bsize=32, k=10, nY=2, pk=10):
    adj = tf.placeholder(tf.float32, [1,bsize,nV,pk*pk*dim_e])
    nnlbl = tf.placeholder(tf.float32, [1,bsize,nV,pk*dim_f])
    att = tf.placeholder(tf.float32, [1,bsize,nV,pk*dim_a])
    nbr = tf.placeholder(tf.int32, [1,bsize,nV,k]) # |V| x k
    label = tf.placeholder(tf.float32, [bsize,nY])
    n1, n2, n3, n4, n5, n6, f1, remain_rate = 128, 128, 128, 128, 128, 128, 256, 0.8
    cv1_theta = [tf.Variable(xavier_init([pk*(pk*dim_e+dim_f+dim_a), n1])), tf.Variable(tf.zeros(shape=[n1]))]
    cv2_theta = [tf.Variable(xavier_init([n1*k, n2])), tf.Variable(tf.zeros(shape=[n2]))]
    cv3_theta = [tf.Variable(xavier_init([n2*k, n3])), tf.Variable(tf.zeros(shape=[n3]))]
    cv4_theta = [tf.Variable(xavier_init([n3*k, n4])), tf.Variable(tf.zeros(shape=[n4]))]
    cv5_theta = [tf.Variable(xavier_init([n4*k, n5])), tf.Variable(tf.zeros(shape=[n5]))]
    cv6_theta = [tf.Variable(xavier_init([n5*k, n6])), tf.Variable(tf.zeros(shape=[n6]))]
    fc1_theta = [tf.Variable(xavier_init([nV*n6, f1])), tf.Variable(tf.zeros(shape=[f1]))]
    fc2_theta = [tf.Variable(xavier_init([f1, nY])), tf.Variable(tf.zeros(shape=[nY]))]
    
    params = cv1_theta+cv2_theta+cv3_theta+cv4_theta+cv5_theta+cv6_theta+fc1_theta+fc2_theta
    
    x = tf.concat([adj, nnlbl, att], 3)
    # Patchy-San
    x = g_conv_bn(x, cv1_theta, [1,bsize,nV,n1])
    # Ego-Convolution
    x = tf.reshape(aggregate_nbr(nbr,x,k), [1,bsize,nV,k*n1])
    x = g_conv_bn(x, cv2_theta, [1,bsize,nV,n2])
    x = tf.reshape(aggregate_nbr(nbr, x, k), [1,bsize,nV,k*n2])
    x = g_conv_bn(x, cv3_theta, [1,bsize,nV,n3])
    x = tf.reshape(aggregate_nbr(nbr, x, k), [1,bsize,nV,k*n3])
    x = g_conv_bn(x, cv4_theta, [1,bsize,nV,n4])
    x = tf.reshape(aggregate_nbr(nbr,x,k), [1,bsize,nV,k*n4])
    x = g_conv_bn(x, cv5_theta, [1,bsize,nV,n5])
    x = tf.reshape(aggregate_nbr(nbr,x,k), [1,bsize,nV,k*n5])
    x = g_conv_bn(x, cv6_theta, [1,bsize,nV,n6])
    # fc
    x = tf.reshape(tf.transpose(x, [1,0,2,3]), [bsize,nV*n6])
    x = tf.nn.dropout(x, remain_rate)
    x = tf.nn.relu(tf.matmul(x, fc1_theta[0]) + fc1_theta[1])
    x = tf.nn.dropout(x, remain_rate)
    out = tf.matmul(x, fc2_theta[0]) + fc2_theta[1]
    return adj, nbr, nnlbl, att, label, out, params
def model_2L(nV, dim_f, dim_a, dim_e, bsize=32, k=10, nY=2, pk=10):
    adj = tf.placeholder(tf.float32, [1,bsize,nV,pk*pk*dim_e])
    nnlbl = tf.placeholder(tf.float32, [1,bsize,nV,pk*dim_f])
    att = tf.placeholder(tf.float32, [1,bsize,nV,pk*dim_a])
    nbr = tf.placeholder(tf.int32, [1,bsize,nV,k]) # |V| x k
    label = tf.placeholder(tf.float32, [bsize,nY])
    n1, n2, f1, remain_rate = 128, 128, 256, 0.8
    cv1_theta = [tf.Variable(xavier_init([pk*(pk*dim_e+dim_f+dim_a), n1])), tf.Variable(tf.zeros(shape=[n1]))]
    cv2_theta = [tf.Variable(xavier_init([n1*k, n2])), tf.Variable(tf.zeros(shape=[n2]))]
    fc1_theta = [tf.Variable(xavier_init([nV*n2, f1])), tf.Variable(tf.zeros(shape=[f1]))]
    fc2_theta = [tf.Variable(xavier_init([f1, nY])), tf.Variable(tf.zeros(shape=[nY]))]
    
    params = cv1_theta+cv2_theta+fc1_theta+fc2_theta
    
    x = tf.concat([adj, nnlbl, att], 3)
    # Patchy-San
    x = g_conv_bn(x, cv1_theta, [1,bsize,nV,n1])
    # Ego-Convolution
    x = tf.reshape(aggregate_nbr(nbr,x,k), [1,bsize,nV,k*n1])
    x = g_conv_bn(x, cv2_theta, [1,bsize,nV,n2])
    # fc
    x = tf.reshape(tf.transpose(x, [1,0,2,3]), [bsize,nV*n2])
    x = tf.nn.dropout(x, remain_rate)
    x = tf.nn.relu(tf.matmul(x, fc1_theta[0]) + fc1_theta[1])
    x = tf.nn.dropout(x, remain_rate)
    out = tf.matmul(x, fc2_theta[0]) + fc2_theta[1]
    return adj, nbr, nnlbl, att, label, out, params
def model_6L_SF(nV, dim_f, dim_a, dim_e, bsize=32, k=10, nY=2, pk=10):
    adj = tf.placeholder(tf.float32, [1,bsize,nV,pk*pk*dim_e])
    nnlbl = tf.placeholder(tf.float32, [1,bsize,nV,pk*dim_f])
    att = tf.placeholder(tf.float32, [1,bsize,nV,pk*dim_a])
    nbr = tf.placeholder(tf.int32, [1,bsize,nV,k]) # |V| x k
    label = tf.placeholder(tf.float32, [bsize,nY])
    n1, n2, f1, remain_rate = 128, 128, 256, 0.8
    cv1_theta = [tf.Variable(xavier_init([pk*(pk*dim_e+dim_f+dim_a), n1])), tf.Variable(tf.zeros(shape=[n1]))]
    cv2_theta = [tf.Variable(xavier_init([n1*k, n2])), tf.Variable(tf.zeros(shape=[n2]))]
    fc1_theta = [tf.Variable(xavier_init([nV*n2, f1])), tf.Variable(tf.zeros(shape=[f1]))]
    fc2_theta = [tf.Variable(xavier_init([f1, nY])), tf.Variable(tf.zeros(shape=[nY]))]
    
    params = cv1_theta+cv2_theta+fc1_theta+fc2_theta
    
    x = tf.concat([adj, nnlbl, att], 3)
    # Patchy-San
    x = g_conv_bn(x, cv1_theta, [1,bsize,nV,n1])
    # Ego-Convolution
    x = tf.reshape(aggregate_nbr(nbr,x,k), [1,bsize,nV,k*n1])
    x = g_conv_bn(x, cv2_theta, [1,bsize,nV,n2])
    x = tf.reshape(aggregate_nbr(nbr, x, k), [1,bsize,nV,k*n2])
    x = g_conv_bn(x, cv2_theta, [1,bsize,nV,n2])
    x = tf.reshape(aggregate_nbr(nbr, x, k), [1,bsize,nV,k*n2])
    x = g_conv_bn(x, cv2_theta, [1,bsize,nV,n2])
    x = tf.reshape(aggregate_nbr(nbr,x,k), [1,bsize,nV,k*n2])
    x = g_conv_bn(x, cv2_theta, [1,bsize,nV,n2])
    x = tf.reshape(aggregate_nbr(nbr,x,k), [1,bsize,nV,k*n2])
    x = g_conv_bn(x, cv2_theta, [1,bsize,nV,n2])
    # fc
    x = tf.reshape(tf.transpose(x, [1,0,2,3]), [bsize,nV*n2])
    x = tf.nn.dropout(x, remain_rate)
    x = tf.nn.relu(tf.matmul(x, fc1_theta[0]) + fc1_theta[1])
    x = tf.nn.dropout(x, remain_rate)
    out = tf.matmul(x, fc2_theta[0]) + fc2_theta[1]
    return adj, nbr, nnlbl, att, label, out, params
def model_3L(nV, dim_f, dim_a, dim_e, bsize=32, k=10, nY=2, pk=10): # for visualization
    adj = tf.placeholder(tf.float32, [1,bsize,nV,pk*pk*dim_e])
    nnlbl = tf.placeholder(tf.float32, [1,bsize,nV,pk*dim_f])
    att = tf.placeholder(tf.float32, [1,bsize,nV,pk*dim_a])
    nbr = tf.placeholder(tf.int32, [1,bsize,nV,k]) # |V| x k
    label = tf.placeholder(tf.float32, [bsize,nY])
    n1, n2, n3, f1, remain_rate = 128, 128, 128, 256, 0.8
    cv1_theta = [tf.Variable(xavier_init([pk*(pk*dim_e+dim_f+dim_a), n1])), tf.Variable(tf.zeros(shape=[n1]))]
    cv2_theta = [tf.Variable(xavier_init([n1*k, n2])), tf.Variable(tf.zeros(shape=[n2]))]
    cv3_theta = [tf.Variable(xavier_init([n2*k, n3])), tf.Variable(tf.zeros(shape=[n3]))]
    fc1_theta = [tf.Variable(xavier_init([nV*n3, f1])), tf.Variable(tf.zeros(shape=[f1]))]
    fc2_theta = [tf.Variable(xavier_init([f1, nY])), tf.Variable(tf.zeros(shape=[nY]))]
    
    params = cv1_theta+cv2_theta+cv3_theta+fc1_theta+fc2_theta
    
    x = tf.concat([adj, nnlbl, att], 3)
    # Patchy-San
    x = g_conv_bn(x, cv1_theta, [1,bsize,nV,n1])
    # Ego-Convolution
    x = tf.reshape(aggregate_nbr(nbr,x,k), [1,bsize,nV,k*n1])
    x = g_conv_bn(x, cv2_theta, [1,bsize,nV,n2])
    x = tf.reshape(aggregate_nbr(nbr, x, k), [1,bsize,nV,k*n2])
    x = g_conv_bn(x, cv3_theta, [1,bsize,nV,n3])
    # fc
    x = tf.reshape(tf.transpose(x, [1,0,2,3]), [bsize,nV*n3])
    x = tf.nn.dropout(x, remain_rate)
    x = tf.nn.relu(tf.matmul(x, fc1_theta[0]) + fc1_theta[1])
    x = tf.nn.dropout(x, remain_rate)
    out = tf.matmul(x, fc2_theta[0]) + fc2_theta[1]
    return adj, nbr, nnlbl, att, label, out, params
def cv(name, lr=1e-3, bsize=32, max_epch=1000, patience=10, k=3, pk=10, gpu_frac=0.99, loop=5, FOLD=10, model_type='6L'):
    # load data
    nbrs = cPickle.load(open('{}/{}/{}-{}-conv.pkl'.format(DATASET_DIR, PROC_DIR, name, k), 'rb')) if 'Conv' in model_type else cPickle.load(open('{}/{}/{}-{}.pkl'.format(DATASET_DIR, PROC_DIR, name, k), 'rb'))
    adjs = cPickle.load(open('{0}/{1}/{2}-{3}x{3}.pkl'.format(DATASET_DIR, PROC_DIR, name, pk), 'rb'))
    nnlbls = cPickle.load(open('{}/{}/{}-{}-nnlabel.pkl'.format(DATASET_DIR, PROC_DIR, name, pk), 'rb'))
    atts = cPickle.load(open('{}/{}/{}-{}-att.pkl'.format(DATASET_DIR, PROC_DIR, name, pk), 'rb'))
    nlabels = cPickle.load(open('{}/{}/{}-nlabels.pkl'.format(DATASET_DIR, PROC_DIR, name), 'rb'))
    elabels = cPickle.load(open('{}/{}/{}-elabels.pkl'.format(DATASET_DIR, PROC_DIR, name), 'rb'))
    labels = cPickle.load(open('{}/{}/{}-label.pkl'.format(DATASET_DIR, PROC_DIR, name), 'rb'))
    Y_map = {c:i for i,c in enumerate(sorted(list(set(labels))))}
    N_map = {c:i for i,c in enumerate(sorted(list(set(nlabels))))}
    E_map = {c:i for i,c in enumerate(sorted(list(set(elabels))))}
    print(E_map)
    N, dim_f, dim_a, dim_e = len(labels), len(N_map), atts[0][0].shape[1] if atts[0][0] is not None else 0, len(E_map)
    dim_f = 0 if dim_f == 1 else dim_f
    dim_f = 0 if 'Compound' in name else dim_f
    nV = max([len(nbr) for nbr in nbrs])
    print('max #node={}, #nlabel={}, #elabel={}, #class={}'.format(nV, dim_f, dim_e, len(Y_map)))
    nY = len(set(labels))
    # define model
    st = time.time()
    # adjust batch size
    bsize = min(bsize, int((N+FOLD-1)/FOLD)-1)
    X_adj, X_nbr, X_nn, X_att, Y_gt, Y_logit, params = build_model(nV+1, dim_f, dim_a, dim_e, bsize=bsize, nY=nY, k=k, pk=pk, model_type=model_type.split('-')[-1])
    print('[compile] takes {:.2f}s'.format(time.time()-st))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Y_logit, labels=Y_gt))
    pred_fn = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y_gt, 1), tf.argmax(tf.sigmoid(Y_logit), 1)), tf.float32))
    opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
    solver = opt.minimize(loss, var_list=params)
    # define weight assign op
    var_placeholders = var_placeholders = [tf.placeholder(tf.float32, shape=p.shape) for p in params]
    assign_op = [v.assign(p) for (v, p) in zip(params, var_placeholders)]
    # init session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction=gpu_frac
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    params_val = sess.run(params)
    # 10-fold CV
    order = np.random.permutation(N)
    cv_size = int((N+FOLD-1)/FOLD)
    cv_acc, cv_vacc = [], []
    h_accs, h_vaccs = [], []
    bparams, times = [], []
    test_accs, best_test = [], 0
    dump = {}
    for cv_i in range(FOLD):
        acc_th, bvacc, bparam = 0.99, 0, None
        for re in range(loop):
            st = time.time()
            # init weight
            sess.run([assign_op], feed_dict={v:p for v,p in zip(var_placeholders, [xavier_init_val(p.shape) for p in params_val])})
            # train:test=9:1
            test_cur = order[cv_i*cv_size:(cv_i+1)*cv_size]
            train_all = [x for x in range(N) if x not in test_cur]
            # train:val=9:1
            torder = np.random.permutation(train_all)
            tv_split = int(len(train_all)*0.9)
            train_cur, val_cur = torder[:tv_split], torder[tv_split:]
            train_nbrs, train_adjs, train_A, train_N, train_Y = [], [], [], [], []
            for x in train_cur:
                train_Y.append(labels[x]) 
                train_N.append(nnlbls[x])
                train_A.append(atts[x])
                train_nbrs.append([nbrs[x]])
                train_adjs.append([adjs[x]])
            val_nbrs, val_adjs, val_A, val_N, val_Y = [], [], [], [], []
            for x in val_cur:
                val_Y.append(labels[x])
                val_N.append(nnlbls[x])
                val_A.append(atts[x])
                val_nbrs.append([nbrs[x]])
                val_adjs.append([adjs[x]]) 
            print((len(train_Y), len(val_Y), len(test_cur)))
            train_nbrs, train_adjs, train_N, train_A, train_Y = np.array(train_nbrs), np.array(train_adjs), np.array(train_N), np.array(train_A), np.array(train_Y)
            val_nbrs, val_adjs, val_N, val_A, val_Y = np.array(val_nbrs), np.array(val_adjs), np.array(val_N), np.array(val_A), np.array(val_Y)
            train_gen = batch_generator(train_adjs, train_nbrs, train_N, train_A, train_Y, N_map, E_map, Y_map, V_num=nV, bsize=bsize, dim_f=dim_f, dim_a=dim_a, dim_e=dim_e, nY=nY, k=k, pk=pk)        
            print('[{}-th CV]'.format(cv_i))
            # training
            it, epch, c, best, lparam = 0, 0, 0, 0, None
            eaccs, evaccs, taccs, tvaccs = [], [], [], []
            while True:
                X_mb = next(train_gen)
                _, acc = sess.run([solver, pred_fn], feed_dict={X_adj:X_mb[0], X_nbr:X_mb[1], X_nn:X_mb[2], X_att:X_mb[3], Y_gt:X_mb[4]})
                if X_mb[5] == epch:
                    taccs.append(acc)
                    del X_mb
                else:
                    epch = X_mb[5]
                    val_gen = batch_generator(val_adjs, val_nbrs, val_N, val_A, val_Y, N_map, E_map, Y_map, V_num=nV, bsize=bsize, dim_f=dim_f, dim_a=dim_a, dim_e=dim_e, nY=nY, k=k, pk=pk)
                    while True:
                        X_val = next(val_gen)
                        val_acc = sess.run([pred_fn], feed_dict={X_adj:X_val[0], X_nbr:X_val[1], X_nn:X_val[2], X_att:X_mb[3], Y_gt:X_val[4]})[0]
                        tvaccs.append(val_acc)
                        if X_val[5] > 0:
                            break
                        del X_val
                    macc, mvacc = np.mean(taccs), np.mean(tvaccs)
                    print('{}th epoch: acc={:.3f} | val_acc={:.3f}'.format(epch, macc, mvacc))
                    eaccs.append(macc)
                    evaccs.append(mvacc)
                    taccs, tvaccs = [], []
                    if best > mvacc:
                        c += 1
                    else:
                        best, c, lparam = mvacc, 0, sess.run(params)
                    if c > patience or epch > max_epch or (1.0-mvacc)<1e-5:
                        break
                it += 1
            mi = np.argmax(evaccs)
            v = evaccs[mi]
            if v > bvacc:
                bacc, bvacc, baccs, bvaccs, border = eaccs[mi], v, eaccs, evaccs, order
                bparam = lparam
                ti = time.time()-st
                print('take time = {:.3f}s'.format(ti))
        # test acc
        sess.run([assign_op], feed_dict={v:p for v,p in zip(var_placeholders, bparam)})
        tnbrs, tadjs, tN, tA, tY = [], [], [], [], []
        for x in test_cur:
            tN.append(nnlbls[x])
            tA.append(atts[x])
            tnbrs.append([nbrs[x]])
            tadjs.append([adjs[x]])
            tY.append(labels[x])
        tnbrs, tadjs, tN, tA, tY = np.array(tnbrs), np.array(tadjs), np.array(tN), np.array(tA), np.array(tY)
        tgen = batch_generator(tadjs, tnbrs, tN, tA, tY, N_map, E_map, Y_map, V_num=nV, bsize=min(bsize, len(tY)-1), dim_f=dim_f, dim_a=dim_a, dim_e=dim_e, nY=nY, k=k, pk=pk)
        taccs = []
        while True:
            tX = next(tgen)
            tacc = sess.run([pred_fn], feed_dict={X_adj:tX[0], X_nbr:tX[1], X_nn:tX[2], X_att:tX[3], Y_gt:tX[4]})[0]
            taccs.append(tacc)
            if tX[5] > 0:
                break
        tacc = np.mean(taccs)
        print('[{}th CV] test_acc={:.3f} | val_acc={:.3f}'.format(cv_i, tacc, bvacc))
        test_accs.append(tacc)
        ti = time.time()-st
        times.append(ti)
        cv_acc.append(bacc)
        cv_vacc.append(bvacc)
        h_accs.append(baccs)
        h_vaccs.append(bvaccs)
        if best_test < tacc:
            bparams = bparam
        dump['cv_acc'] = cv_acc
        dump['cv_vacc'] = cv_vacc
        dump['accs'] = h_accs
        dump['vaccs'] = h_vaccs
        dump['split'] = order
        dump['time'] = times
        dump['params'] = bparams
        dump['test_acc'] = test_accs
        cPickle.dump(dump, open('{}-k{}-{}_cv_hist.pkl'.format(name, k, model_type), 'wb'))
    cPickle.dump(dump, open('{}-k{}-{}_cv_hist.pkl'.format(name, k, model_type), 'wb'))
    sess.close()
    print('[10-Fold CV] val_acc={:.3f}, test_acc={:.3f}, time={:.3f}s for model {}'.format(np.mean(cv_vacc), np.mean(test_accs), np.mean(times), model_type))
    return dump

# parse argument
opt = parse_arg()
os.environ["CUDA_VISIBLE_DEVICES"]='{}'.format(opt.gpu_id)
cv(opt.dataset, lr=float(opt.lr), patience=10, k=int(opt.K), pk=int(opt.PK), bsize=int(opt.bsize), gpu_frac=float(opt.gpu_frac), loop=int(opt.loop), FOLD=int(opt.FOLD), model_type=opt.model_type)