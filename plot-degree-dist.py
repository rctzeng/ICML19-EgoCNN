import _pickle as cPickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

"""
Scale-Free Networks found on Reddit
    => plot the degree distribution in log-log scale 
    => follows a power-law
"""

Gs = cPickle.load(open('../dataset/proc/REDDIT-BINARY-Gs.pkl', 'rb'))
degs, mdeg = [], 0
for G in Gs:
    t = list(nx.degree(G).values())
    degs += t
    if len(t) > 0:
        mdeg = max(max(t), mdeg)
degree_seq = sorted(np.array(degs).reshape((-1)), reverse=True)
n, bins, patches = plt.hist(degree_seq, 100, log=True)
xs = [(bins[i]+bins[i+1])/2 for i in range(len(n))]
plt.figure(figsize=(5,4))
plt.scatter(np.log(xs) ,np.log(n) , c='black')
plt.title("degree rank plot")
plt.ylabel("# of nodes (log)")
plt.xlabel("degree (log)")
plt.savefig('reddit-scale-free.png')