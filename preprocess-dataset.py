import os
import optparse
import _pickle as cPickle
import numpy as np
import networkx as nx

"""
Extract labeled neighborhood graphs
--------------------------------------
 * k neighbors
     (option-1) select neighbors from only 1-hop neighbors or upto k by BFS?
     (option-2) select according to 1-WL[occurence of 1-WL labels] or degree?
     (option-3) assign smaller or larger value of (option-1) higher priority?
 * node labels
 * edge labels
 * node attributes
"""
GDIR = 'proc'
DATASET_DIR = 'dataset'
DATASET_LIST = ['MUTAG', 'PTC_MR', 'PROTEINS', 'NCI1', 'DD', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINAY', 'COLLAB', 'Compound_Alk-Alc', 'Compound_Asym-Sym']

def parse_arg():
    parser = optparse.OptionParser()
    parser.add_option('-n', dest='dataset', help='specify the name of dataset in one of {}'.format(DATASET_LIST))
    parser.add_option('-k', dest='K', help='the #neighbor in the local neighborhoods')
    parser.add_option('-s', dest='sort_vertex', action="store_true", default=False, help='sort all vertex before processing? [Y-general, N-visualization]')
    (options, args) = parser.parse_args()
    return options
def G_to_NX_sparse(X, Y):
    """convert sparse adj matrix to NetworkX Graph"""
    Gs = []
    N = len(Y)
    for n in range(N):
        x = X[n]
        G = nx.DiGraph()
        for i,j,w in x:
            G.add_edge(i,j, weight=w)
        Gs.append(G)
    return Gs, Y
def gen_compound(name, N=50, nP=10):
    """
    (1) Alkane vs Alcohol
        N: the #carbon atom in compound, ex: N=50 generates compounds of different length from 1 to 50 carbons
        P: # of permutation to relabeling the vertex order for each generated compound
    ----------------------------------------------
    (2) Asymmetric Isomer vs Symmetric Isomer
        N: the #carbon atom in compound, ex: N=50 generates compounds of different length from 1 to 50 carbons
        P: # of permutation to relabeling the vertex order for each generated compound
    """
    def gen_alcohol(nC): # C_n H_2n+1 OH
        G, nlabel = nx.Graph(), {}
        for i in range(nC):
            c = i*3+1
            G.add_edge(c,c+1,weight=1)
            G.add_edge(c,c+2,weight=1)
            nlabel[c] = 'C'
            nlabel[c+1] = 'H'
            nlabel[c+2] = 'H'
            if i == 0:
                G.add_edge(c,c-1,weight=1)
                nlabel[c-1] = 'H'
            else:
                G.add_edge(c,c-3,weight=1)
                if i == nC-1:
                    G.add_edge(c,c+3,weight=1)
                    G.add_edge(c+3,c+4,weight=1)
                    nlabel[c+3] = 'O'
                    nlabel[c+4] = 'H'
        return G, nlabel
    def gen_alkane(nC): # C_n H_2n+2
        G, nlabel = nx.Graph(), {}
        for i in range(nC):
            c = i*3+1
            G.add_edge(c,c+1,weight=1)
            G.add_edge(c,c+2,weight=1)
            nlabel[c] = 'C'
            nlabel[c+1] = 'H'
            nlabel[c+2] = 'H'
            if i == 0:
                G.add_edge(c,c-1,weight=1)
                nlabel[c-1] = 'H'
            else:
                G.add_edge(c,c-3,weight=1)
                if i == nC-1:
                    G.add_edge(c,c+3,weight=1)
                    nlabel[c+3] = 'H'
        return G, nlabel
    def gen_asym(nC):
        G, nlabel = nx.Graph(), {}
        cc = np.random.randint(nC)
        nlabel[nC*2+1] = 'C'
        for i in range(nC*2+1):
            nlabel[i] = 'C'
            if i > 0:
                G.add_edge(i,i-1,weight=1)
            if i == cc:
                G.add_edge(i,nC*2+1,weight=1)
        return G, nlabel
    def gen_sym(nC):
        G, nlabel = nx.Graph(), {}
        nlabel[nC*2+1] = 'C'
        for i in range(nC*2+1):
            nlabel[i] = 'C'
            if i > 0:
                G.add_edge(i,i-1,weight=1)
            if i == nC:
                G.add_edge(i,nC*2+1,weight=1)
        return G, nlabel
    def permute(G, nlabel):
        A = nx.adjacency_matrix(G).todense()
        N = A.shape[0]
        nids = list(G.nodes())
        order = np.random.permutation(nids)
        op = {nid:i for i,nid in enumerate(nids)}
        mp = {nid:i for i,nid in enumerate(order)}
        mm = {nid:nids[i] for i,nid in enumerate(order)}
        rA = np.zeros_like(A)
        for i in range(N):
            for j in range(N):
                rA[i,j] = A[mp[nids[i]],mp[nids[j]]]
        rnlabel = {mm[nid]:nlabel[mm[nid]] for nid in nids}
        rG = nx.from_numpy_matrix(rA)
        return rG, rnlabel
    cls = name.split('-')
    Gs, Ys, nlabels = [], [], []
    
    if name == 'Asym-Sym':
        for i in range(N):
            G, nlabel = gen_asym(5+i)
            Gs.append(G)
            Ys.append(0)
            nlabels.append(nlabel)
            for p in range(nP-1):
                pG, pL = permute(G, nlabel)
                Gs.append(G)
                Ys.append(0)
                nlabels.append(pL)
            G, nlabel = gen_sym(5+i)
            Gs.append(G)
            Ys.append(1)
            nlabels.append(nlabel)
            for p in range(nP-1):
                pG, pL = permute(G, nlabel)
                Gs.append(G)
                Ys.append(1)
                nlabels.append(pL)
    elif name == 'Alk-Alc':
        for n in range(1,N):
            for icl,fn in enumerate([gen_alkane, gen_alcohol]):
                G, nlabel = fn(2*n)
                Gs.append(G)
                Ys.append(icl)
                nlabels.append(nlabel)
                for p in range(nP-1):
                    pG, pL = permute(G, nlabel)
                    Gs.append(G)
                    Ys.append(icl)
                    nlabels.append(pL)
    if not os.path.exists('{}/{}'.format(DATASET_DIR, name)):
        os.makedirs('{}/{}'.format(DATASET_DIR, name))
    cPickle.dump(Gs, open('{}/{}/N{}-P{}-Gs.pkl'.format(DATASET_DIR, name, N, nP), 'wb'))
    cPickle.dump(nlabels, open('{}/{}/N{}-P{}-nlabels.pkl'.format(DATASET_DIR, name, N, nP), 'wb'))
    cPickle.dump(Ys, open('{}/{}/N{}-P{}-labels.pkl'.format(DATASET_DIR, name, N, nP), 'wb'))
def read_G_dataset(name):
    """
    loads graph classification dataset
    ---------------------------
    returns [NetworkX Gs, graph labels]
    """
    if not os.path.exists('{}/{}/{}-Gs.pkl'.format(DATASET_DIR, GDIR, name)):
        if name in ['MUTAG', 'PTC_MR', 'PROTEINS', 'NCI1', 'NCI109', 'ENZYMES', 'DD',
            'COLLAB', 'REDDIT-BINARY', 'IMDB-BINARY', 'IMDB-MULTI']:
            with open('{0}/{1}/{1}_graph_labels.txt'.format(DATASET_DIR, name), 'r') as f:
                data = f.readlines()
            Y = [(int(line)) for line in data]
            with open('{0}/{1}/{1}_graph_indicator.txt'.format(DATASET_DIR, name), 'r') as f:
                data = f.readlines()
            NG = {i+1:int(data[i]) for i in range(len(data))}
            Fs, nlabels = {}, []
            # node label
            if os.path.exists('{0}/{1}/{1}_node_labels.txt'.format(DATASET_DIR, name)):
                with open('{0}/{1}/{1}_node_labels.txt'.format(DATASET_DIR, name), 'r') as f:
                    for i,line in enumerate(f):
                        nid, gid = i+1, NG[i+1]
                        if gid not in Fs.keys():
                            Fs[gid]={}
                        Fs[gid][nid]=int(line)
                        nlabels.append(int(line))
                Fs = [Fs[k] for k in sorted(list(Fs.keys()))]
            else:
                nlabels = [1]
                Fs = None
            cPickle.dump(nlabels, open('{}/{}/{}-nlabels.pkl'.format(DATASET_DIR, GDIR, name), 'wb'))
            # node attributes
            Atts = {}
            # node label
            if os.path.exists('{0}/{1}/{1}_node_attributes.txt'.format(DATASET_DIR, name)):
                with open('{0}/{1}/{1}_node_attributes.txt'.format(DATASET_DIR, name), 'r') as f:
                    for i,line in enumerate(f):
                        nid, gid = i+1, NG[i+1]
                        if gid not in Atts.keys():
                            Atts[gid]={}
                        Atts[gid][nid]=[float(x) for x in line.split(',')]
                Atts = [Atts[k] for k in sorted(list(Atts.keys()))]
            else:
                Atts = None
            # edge label
            EW = []
            if os.path.exists('{0}/{1}/{1}_edge_labels.txt'.format(DATASET_DIR, name)):
                with open('{0}/{1}/{1}_edge_labels.txt'.format(DATASET_DIR, name), 'r') as f:
                    for line in f:
                        EW.append(int(line)+1)
            else:
                EW = [1]
            cPickle.dump(EW, open('{}/{}/{}-elabels.pkl'.format(DATASET_DIR, GDIR, name), 'wb'))
            X = {}
            with open('{0}/{1}/{1}_A.txt'.format(DATASET_DIR, name), 'r') as f:
                for i,line in enumerate(f):
                    els = line.split(',')
                    a, b = int(els[0]), int(els[1])
                    if NG[a] not in X.keys():
                        X[NG[a]] = []
                    if NG[b] not in X.keys():
                        X[NG[b]] = []
                    w = EW[i] if len(EW)>1 else 1
                    for gid in list(set([NG[a],NG[b]])):
                        X[gid] += [(a,b,w),(b,a,w)]
                    if NG[a] != NG[b]:
                        print('{} and {} cross graphs'.format(a, b))
            X = [X[k] for k in sorted(list(X.keys()))]
            Gs, Y = G_to_NX_sparse(X, Y)
        elif 'Compound' in name:
            els = name.split('_')
            N, permute = 50, 10
            if not os.path.exists('{}/{}/N{}-P{}-nlabels.pkl'.format(DATASET_DIR, els[1], N, permute)):
                gen_compound(els[1], N, permute)
            Gs, Fs, Y, Atts = cPickle.load(open('{}/{}/N{}-P{}-Gs.pkl'.format(DATASET_DIR, els[1], N, permute), 'rb')), cPickle.load(open('{}/{}/N{}-P{}-nlabels.pkl'.format(DATASET_DIR, els[1], N, permute), 'rb')), cPickle.load(open('{}/{}/N{}-P{}-labels.pkl'.format(DATASET_DIR, els[1], N, permute), 'rb')), None
            nlabels = [x for F in Fs for x in F.values()]
            cPickle.dump(nlabels, open('{}/{}/{}-nlabels.pkl'.format(DATASET_DIR, GDIR, name), 'wb'))
            EW = [1]
            cPickle.dump(EW, open('{}/{}/{}-elabels.pkl'.format(DATASET_DIR, GDIR, name), 'wb'))
        else:
            raise Exception('{} undefined'.format(name))
    else:
        Gs, Y, Fs, Atts = cPickle.load(open('{}/{}/{}-Gs.pkl'.format(DATASET_DIR, GDIR, name), 'rb')), cPickle.load(open('{}/{}/{}-label.pkl'.format(DATASET_DIR, GDIR, name), 'rb')), cPickle.load(open('{}/{}/{}-Fs.pkl'.format(DATASET_DIR, GDIR, name), 'rb')), cPickle.load(open('{}/{}/{}-Atts.pkl'.format(DATASET_DIR, GDIR, name), 'rb'))
    return Gs, Y, Fs, Atts
def rcpv_fld(G, F, Att, ego, order_dict, idx, i, k=3):
    """
    return node i's
        (1) k neighbors
        (2) neighborhood
        (3) node labels of (1)
    -------------------------------------------------------------
    [parameters]
    G: entire graph
    F: node labels in G
    ego: neighborhoods in G
    order_dict: ordering of neighbors [degree or 1-WL]
    idx: relabeling of node id
    """
    def get_nbr(G, cur_ns):
        """grab 1-hop ahead local neighborhood"""
        nxt_ns = []
        for x in cur_ns:
            nxt_ns += G[x]
        nxt_ns = list(set(nxt_ns))
        return nxt_ns
    ######### selection of neighbors ##########
    # get only 1-hop neighbors
    ns = get_nbr(G, [i])
    """
    # BFS selection of neighbors upto k
    ns = []
    tmp = [i]
    prev = 1
    while len(ns) < k:
        tmp = get_nbr(G, tmp)
        ns += tmp
        ns = list(set(ns))
        if prev == len(ns):
            break
        prev = len(ns)
    """
    ######### neighbor normalization ##########
    # sorting order
    SMALL = True # True: small value first / False: large value first
    SELF = True # True: include self embedding / False: not always
    ns = [idx[x] for x in ns if x in idx.keys()]
    if not SELF:
        ns += [idx[i]]
    ds = [order_dict[x] for x in ns]
    top = np.argsort(ds)
    if not SMALL:
        top = top[::-1]
    nbr = [ns[x] for x in top[:k]] if not SELF else [idx[i]]+[ns[x] for x in top[:k-1]]
    ######### extract node labels of neighbors ##########
    nnlbl = [F[x] for x in nbr]
    ######### extract node attributes of neighbors ##########
    att = np.array([Att[x] for x in nbr]) if Att is not None else None
    ######### extract neighborhoods ##########
    # padding to k neighbors
    while len(nbr) < k:
        nbr += [-1]
    # adj of neighborhood
    adj = np.zeros((k,k))
    ego_i = {(k,v):w for k,v,w in ego[idx[i]]}
    for x in range(k):
        for y in range(k):
            c = ego_i[(nbr[x], nbr[y])] if (nbr[x], nbr[y]) in ego_i.keys() else 0
            adj[x,y] = c
    return nbr, adj, nnlbl, att
def proc_G_dataset(name, k=3, sort_vertex=False):
    def dump(Gs, labels, Fs, Atts, name, k):
        if not os.path.exists('{}/{}/{}-RWL.pkl'.format(DATASET_DIR, GDIR, name)):
            WL, NWL = {}, []
            # calc freq of each label in 1-WL
            for gi in range(len(Gs)):
                G, nwl = Gs[gi], {}
                F = Fs[gi] if Fs is not None else None
                for nid in G.nodes():
                    f_self = F[nid] if F is not None else '1'
                    f_nbr = list(sorted([F[nbr_id] for nbr_id in G.neighbors(nid)])) if F is not None else ['1' for nbr_id in G.neighbors(nid)]
                    agg_nbr = ''.join(str(x) for x in [f_self]+f_nbr)
                    nwl[nid] = agg_nbr
                    if agg_nbr not in WL.keys():
                        WL[agg_nbr] = 0
                    WL[agg_nbr] += 1
                NWL.append(nwl)
            RWL = [{k:WL[v] for k,v in nwl.items()} for nwl in NWL]
        else:
            RWL = cPickle.load(open('{}/{}/{}-RWL.pkl'.format(DATASET_DIR, GDIR, name), 'rb'))
        degs, nbrs, adjs, nnlbls, atts, lbls, cnbrs = [], [], [], [], [], [], []
        for gi in range(len(Gs)):
            G, F, Att = Gs[gi], Fs[gi] if Fs is not None else None, Atts[gi] if Atts is not None else None
            nids, degs = [], []
            nodes = G.nodes()
            ks = F.keys() if F is not None else nodes
            for i in ks:
                nids.append(i)
                degs.append(G.degree(i) if i in nodes else 0)
            if sort_vertex:
                # sort vertex by degree
                idx = {}
                for i,oid in enumerate(np.argsort(degs)[::-1]):
                    idx[nids[oid]] = i
            else:
                idx = {nid:i for i,nid in enumerate(nodes)}
            F = {idx[k]:v for k,v in F.items()} if F is not None else {idx[k]:1 for k in idx.keys()}
            Att = {idx[k]:v for k,v in Att.items()} if Att is not None else None
            ego = {idx[i]:[(idx[k],idx[v],w['weight']) for k,v,w in nx.ego_graph(G, i).edges(data=True)] for i in G.nodes()}
            ########### sorting dictionary for neighbor normalization #################
            deg = {idx[i]:G.degree(i) for i in G.nodes()} # by degree
            WL = {idx[i]:RWL[gi][i] for i in G.nodes()} # by 1-WL
            ORDER = WL
            NBR, ADJ, NNLBL, ATT, CNBR = {}, {}, {}, {}, {}
            node_num = G.order()
            for i in G.nodes():
                nbr, adj, nnlbl, att = rcpv_fld(G, F, Att, ego, ORDER, idx, i, k=k)
                NBR[idx[i]] = nbr
                ADJ[idx[i]] = adj
                NNLBL[idx[i]] = nnlbl
                ATT[idx[i]] = att
                tcnbr = [x for x in range(max(0,idx[i]-int(k/2)), min(node_num,idx[i]+(k-int(k/2))))]
                tcnbr += [-1 for x in range(k-len(tcnbr))]
                CNBR[idx[i]] = tcnbr
            nbrs.append(NBR)
            cnbrs.append(CNBR)
            adjs.append(ADJ)
            nnlbls.append(NNLBL)
            atts.append(ATT)
            lbls.append(labels[gi])
            if gi % 1000 == 0 and gi:
                print('{} done'.format(gi))
        if not os.path.exists('{}/{}/{}-Gs.pkl'.format(DATASET_DIR, GDIR, name)):
            cPickle.dump(Gs, open('{}/{}/{}-Gs.pkl'.format(DATASET_DIR, GDIR, name), 'wb'))
            cPickle.dump(lbls, open('{}/{}/{}-label.pkl'.format(DATASET_DIR, GDIR, name), 'wb'))
            cPickle.dump(Fs, open('{}/{}/{}-Fs.pkl'.format(DATASET_DIR, GDIR, name), 'wb'))
            cPickle.dump(Atts, open('{}/{}/{}-Atts.pkl'.format(DATASET_DIR, GDIR, name), 'wb'))
            cPickle.dump(RWL, open('{}/{}/{}-RWL.pkl'.format(DATASET_DIR, GDIR, name), 'wb'))
        cPickle.dump(nbrs, open('{}/{}/{}-{}.pkl'.format(DATASET_DIR, GDIR, name, k), 'wb'))
        cPickle.dump(cnbrs, open('{}/{}/{}-{}-conv.pkl'.format(DATASET_DIR, GDIR, name, k), 'wb'))
        cPickle.dump(adjs, open('{0}/{1}/{2}-{3}x{3}.pkl'.format(DATASET_DIR, GDIR, name, k), 'wb'))
        cPickle.dump(nnlbls, open('{}/{}/{}-{}-nnlabel.pkl'.format(DATASET_DIR, GDIR, name, k), 'wb'))
        cPickle.dump(atts, open('{}/{}/{}-{}-att.pkl'.format(DATASET_DIR, GDIR, name, k), 'wb'))
    if not os.path.exists('{}/{}'.format(DATASET_DIR, GDIR)):
        os.makedirs('{}/{}'.format(DATASET_DIR, GDIR))
    Gs, labels, Fs, Atts = read_G_dataset(name=name)
    dump(Gs, labels, Fs, Atts, name, k)
    print('dumping training set done')

opt = parse_arg()
proc_G_dataset(opt.dataset, k=int(opt.K), sort_vertex=opt.sort_vertex)