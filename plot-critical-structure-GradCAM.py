import optparse
import os
import time
import _pickle as cPickle
import numpy as np
import networkx as nx
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""
Plot Critical Structure of trained model[Patchy-San + Ego-Convolution]
The critical part in each graph of the specified dataset is plotted in folder `G-[dataset-name]/` in Gexf format
"""


PROC_DIR = 'proc'
DATASET_DIR = 'dataset'
DATASET_LIST = ['MUTAG', 'PTC_MR', 'PROTEINS', 'NCI1', 'DD', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'COLLAB', 'Compound_Alk-Alc', 'Compound_Asym-Sym']


def parse_arg():
    """utility function to parse argument from command line"""
    parser = optparse.OptionParser()
    parser.add_option('-n', dest='dataset', help='specify the name of dataset in one of {}'.format(DATASET_LIST))
    parser.add_option('-g', dest='gpu_id', default='0', help='specify to run on which GPU')
    parser.add_option('-f', dest='gpu_frac', default='0.99', help='specify the memory utilization of GPU')
    parser.add_option('-k', dest='K', default='4', help='hyperparameter k(#neighobr) of Ego-Convolution')
    parser.add_option('-p', dest='PK', default='10', help='hyperparameter k(size of neighbrohood of Patchy-San')
    parser.add_option('-t', dest='th', default='0.8', help='threshold to determine important neighborhoods')
    parser.add_option('-L', dest='nL', default=6, help='#layer(including Patchy-San and Ego-Convolution) of Ego-CNN')
    parser.add_option('-m', dest='model_type', default='6L', help="specify which model to use in one of ['6L', '3L']")
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
        order = range(N)
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
    return tf.stack([[tf.nn.embedding_lookup(fmap, nbr_dict) for b_nbr_dict, b_fmap in zip(tf.unstack(pb_nbr_dict), tf.unstack(pb_fmap)) for nbr_dict, fmap in zip(tf.unstack(b_nbr_dict), tf.unstack(b_fmap))]])
def g_conv_bn(X, theta, shape):
    """
    X: [1, bsize, nV, n1] => [1, bsize*nV, n1] == swap-axis ==> [bsize*nV, n1]
    W: [n1, n2]
    XW: [bsize*nV, 1, n2] == swap-axis => [1, bsize, nV, n2]
    """
    _, bsize, nV, n2 = shape
    X = tf.reshape(tf.transpose(tf.reshape(X, [1, bsize*nV, -1]), [1,0,2]), [bsize*nV, -1])
    return tf.reshape(tf.transpose(tf.reshape(tf.nn.relu(bn(tf.matmul(X, theta[0])+theta[1])), [bsize*nV, 1, n2]), [1,0,2]), [1, bsize, nV, n2])
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
    lastEgo = x
    # fc
    x = tf.reshape(tf.transpose(x, [1,0,2,3]), [bsize,nV*n6])
    x = tf.nn.dropout(x, remain_rate)
    x = tf.nn.relu(tf.matmul(x, fc1_theta[0]) + fc1_theta[1])
    x = tf.nn.dropout(x, remain_rate)
    out = tf.matmul(x, fc2_theta[0]) + fc2_theta[1]
    return adj, nbr, nnlbl, att, label, lastEgo, out, params
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
    lastEgo = x
    # fc
    x = tf.reshape(tf.transpose(x, [1,0,2,3]), [bsize,nV*n3])
    x = tf.nn.dropout(x, remain_rate)
    x = tf.nn.relu(tf.matmul(x, fc1_theta[0]) + fc1_theta[1])
    x = tf.nn.dropout(x, remain_rate)
    out = tf.matmul(x, fc2_theta[0]) + fc2_theta[1]
    return adj, nbr, nnlbl, att, label, lastEgo, out, params
def build_gradCAM(lastEgo, Y_logit, nV, nY=2):
    """Implement Grad-CAM(ICCV'17)"""
    cams = []
    for c in range(nY):
        # calculate the importance of each feature map
        alpha = tf.reduce_sum(tf.gradients(Y_logit[:,c], lastEgo)[0], axis=(1,2))
        # normalize alpha
        alpha = tf.reshape(alpha/tf.reduce_sum(alpha, axis=1), (-1,1))
        cam = tf.transpose(tf.reshape(alpha*tf.reshape(tf.transpose(lastEgo, [0,1,3,2]), (-1,nV+1)), (-1,128,nV+1)), [0,2,1])
        # linear combine the feature map to generate CAM
        cams.append(tf.nn.relu(cam))
    cams = tf.stack(cams, axis=3)
    return cams

def get_model(model_type='6L'):
    if model_type == '3L':
        return model_3L
    return model_6L
############################################ plot critical structures ############################################
def plot_critical(name, th, nL=6, bsize=1, k=4, pk=4, gpu_frac=0.99, psize=(250,400), model_type='6L'):
    """Visualization for Trained Model"""
    def get_nbr(NBR, cur_ns):
        nxt_ns = []
        for x in cur_ns:
            nxt_ns += NBR[x].tolist()
        return nxt_ns
    def downF(param, cur_Adj, k=4): # truncated negative activation <= RELU
        n = cur_Adj.shape[0]
        return np.dot(np.minimum(cur_Adj, 0)-param[1], param[0].T).reshape((n*k, -1))
    # load data: for model
    nbrs = cPickle.load(open('{}/{}/{}-{}.pkl'.format(DATASET_DIR, PROC_DIR, name, k), 'rb'))
    adjs = cPickle.load(open('{0}/{1}/{2}-{3}x{3}.pkl'.format(DATASET_DIR, PROC_DIR, name, pk), 'rb'))
    nnlbls = cPickle.load(open('{}/{}/{}-{}-nnlabel.pkl'.format(DATASET_DIR, PROC_DIR, name, pk), 'rb'))
    atts = cPickle.load(open('{}/{}/{}-{}-att.pkl'.format(DATASET_DIR, PROC_DIR, name, pk), 'rb'))
    nlabels = cPickle.load(open('{}/{}/{}-nlabels.pkl'.format(DATASET_DIR, PROC_DIR, name), 'rb'))
    elabels = cPickle.load(open('{}/{}/{}-elabels.pkl'.format(DATASET_DIR, PROC_DIR, name), 'rb'))
    labels = cPickle.load(open('{}/{}/{}-label.pkl'.format(DATASET_DIR, PROC_DIR, name), 'rb'))
    # load data: for visualization
    Gs = cPickle.load(open('{}/{}/{}-Gs.pkl'.format(DATASET_DIR, PROC_DIR, name), 'rb'))
    Y_map = {c:i for i,c in enumerate(sorted(list(set(labels))))}
    N_map = {c:i for i,c in enumerate(sorted(list(set(nlabels))))}
    E_map = {c:i for i,c in enumerate(sorted(list(set(elabels))))}
    if 'Compound' in name:
        els = name.split('_')
        N, P = 50, 10 # default
        nlbls = cPickle.load(open('{}/{}/N{}-P{}-nlabels.pkl'.format(DATASET_DIR, els[1], N, P), 'rb'))
    elif name in ['IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'COLLAB']:
        nlbls = None
    else:
        raise Exception('unimplement')
    
    DIR_PATH, FIG_PATH = 'G-{}/gexf-k{}-{}'.format(name, k, model_type), 'G-{}/fig-k{}-{}'.format(name, k, model_type)
    for path in ['G-{}'.format(name), DIR_PATH, FIG_PATH]:
        if not os.path.exists(path):
            os.makedirs(path)
    
    N, dim_f, dim_a, dim_e = len(labels), len(N_map), atts[0][0].shape[1] if atts[0][0] is not None else 0, len(E_map)
    dim_f = 0 if dim_f == 1 else dim_f
    dim_f = 0 if 'Compound' in name else dim_f
    nV = max([len(nbr) for nbr in nbrs])
    print('max #node={}, #nlabel={}, #elabel={}, #class={}'.format(nV, dim_f, dim_e, len(Y_map)))
    nY = len(set(labels))
    # define model
    st = time.time()
    egonet = get_model(model_type)
    X_adj, X_nbr, X_nn, X_att, Y_gt, lastEgo, Y_logit, params = egonet(nV+1, dim_f, dim_a, dim_e, bsize=1, nY=nY, k=k, pk=pk) 
    pred_fn = tf.cast(tf.equal(tf.argmax(Y_gt, 1), tf.argmax(tf.sigmoid(Y_logit), 1)), tf.float32)
    gcam = build_gradCAM(lastEgo, Y_logit, nV, nY=nY)
    
    # define weight assign op
    var_placeholders = var_placeholders = [tf.placeholder(tf.float32, shape=p.shape) for p in params]
    trained_params = cPickle.load(open('{}-k{}-{}_cv_hist.pkl'.format(name, k, model_type), 'rb'))['params']
    assign_op = [v.assign(p) for (v, p) in zip(params, var_placeholders)]
    # init session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction=gpu_frac
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    params_val = sess.run(params)
    # init weight
    sess.run([assign_op], feed_dict={v:p for v,p in zip(var_placeholders, trained_params)})
    # prepare batch
    all_nbrs, all_adjs, all_A, all_N, all_Y = [], [], [], [], []
    for x in range(N):
        all_Y.append(labels[x])
        all_N.append(nnlbls[x])
        all_A.append(atts[x])
        all_nbrs.append([nbrs[x]])
        all_adjs.append([adjs[x]])
    all_nbrs, all_adjs, all_N, all_A, all_Y = np.array(all_nbrs), np.array(all_adjs), np.array(all_N), np.array(all_A), np.array(all_Y)
    gen = batch_generator(all_adjs, all_nbrs, all_N, all_A, all_Y, N_map, E_map, Y_map, V_num=nV, bsize=bsize, dim_f=dim_f, dim_a=dim_a, dim_e=dim_e, nY=nY, k=k, pk=pk)
    ################################## Get Scores and Feature Maps ##################################
    st = time.time()
    Accs, Nbrs, CAMs = [], [], []
    while True:
        X_cur = next(gen)
        if X_cur[5] > 0:
            break
        acc, cam = sess.run([pred_fn, gcam], feed_dict={X_adj:X_cur[0], X_nbr:X_cur[1], X_nn:X_cur[2], X_att:X_cur[3], Y_gt:X_cur[4]})
        Accs += acc.reshape((-1)).tolist()
        Nbrs.append(X_cur[1])
        CAMs.append(cam)
    sess.close()
    CAMs, Nbrs = np.array(CAMs).reshape((-1,nV+1,128,nY)), np.array(Nbrs).reshape((-1, nV+1, pk))
    print('[plot critical] time={:.3f}s'.format(np.mean(time.time()-st)))
    ################################## plot in Matplotlib and Gexf(3D) ##################################
    for gid in range(len(Accs)):
        G, Nbr, CAM, Y = Gs[gid].to_undirected(), Nbrs[gid], CAMs[gid].reshape((nV+1,128,nY)), labels[gid]
        nlbl, CAM, gnodes = {}, CAM[:,:,Y], G.nodes()
        for i,nid in enumerate(gnodes):
            nlbl[nid] = nlbls[gid][nid] if nlbls is not None else '1'
        ###### origin graph ######
        # gexf
        newG = nx.MultiGraph()
        for nid in gnodes:
            newG.add_node(nid, label=nlbl[nid], color='white', size=psize[0])
        for a,b in G.edges():
            newG.add_edge(a, b, color='black', size=1)
        # png (matplotlib)
        pos = nx.spring_layout(G)
        plt.figure(figsize=(10,5))
        if nlbls is None: # for 'REDDIT-BINARY', 'Compound_Asym-Sym'
            nx.draw_networkx(G, pos, with_labels=False)
        else: # for 'Compound_Alk-Alc'
            nx.draw_networkx(G, pos, labels=nlbl)
        nx.draw_networkx_nodes(G, pos=pos, nodelist=list(gnodes), node_color='red', label='-', node_size=300)
        ################## interpolate important neighborhoods by Deconvolution ################
        if True:
            nmap = {i+1:nid for i,nid in enumerate(gnodes)}
            edges = []
            for i,h in enumerate(CAM[1:]):
                cur_ns = [i+1]
                for l in range(nL):
                    cur_ns = get_nbr(Nbr, cur_ns)
                nbrId = np.array(cur_ns).reshape((-1,k))
                
                cur_Adj = h.reshape((-1, 128))
                for l in reversed(range(nL)):
                    param = trained_params[l*2:(l+1)*2]
                    cur_Adj = downF(param, cur_Adj, k=k)
                cur_Adj = cur_Adj.reshape((-1, trained_params[0].shape[0]))[:,:k*k]

                hasLink = cur_Adj>0
                for nid in range(hasLink.shape[0]):
                    for a in range(k):
                        for b in range(a+1,k):
                            if hasLink[nid][a*k+b] or hasLink[nid][b*k+a]:
                                ss, tt = nbrId[nid][a], nbrId[nid][b]
                                if ss in nmap.keys() and tt in nmap.keys():
                                    s, t = nmap[ss], nmap[tt]
                                    if (s,t) in G.edges():
                                        w = max(cur_Adj[nid][a*k+b], cur_Adj[nid][b*k+a])
                                        edges.append((s,t,w,l))
            ###### draw ######
            # gexf
            for a,b,w,l in edges:
                newG.add_edge(a, b, color='grey', weight=5+(int(w*5))**2, label="{}'s".format(l))
            es, ws = [], []
            for a,b,w,l in edges:
                es.append((a,b))
                ws.append(5+(int(w*5))**2)
            nx.draw_networkx_edges(G, pos=pos, edgelist=es, edge_color='lime', width=ws)
        plt.axis('off')
        lgd = plt.legend(loc="upper left", bbox_to_anchor=(1,1))
        plt.savefig('{}/G{}-{}-{}.png'.format(FIG_PATH, gid, Y, th), bbox_extra_artists=(lgd,), bbox_inches='tight')
        # save in GEXF format
        nx.write_gexf(newG, path='{}/G{}-{}-{:.1f}.gexf'.format(DIR_PATH, gid, Y, th))
# parse argument
opt = parse_arg()
os.environ["CUDA_VISIBLE_DEVICES"]='{}'.format(opt.gpu_id)
plot_critical(opt.dataset, th=float(opt.th), nL=int(opt.nL), k=int(opt.K), pk=int(opt.PK), gpu_frac=float(opt.gpu_frac), model_type=opt.model_type)