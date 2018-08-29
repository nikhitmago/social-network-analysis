import sys
import numpy as np
import networkx as nx
import time
from itertools import combinations
from networkx import *
from pyspark import SparkContext

sc = SparkContext(appName="Community")

start = time.time()

inputFile = sys.argv[1]
outputFile = 'Nikhit_Mago_Community.txt'

data = sc.textFile(inputFile)

header = data.first()
users = data.filter(lambda x: x!=header).map(lambda x: x.split(',')).map(lambda x: (int(x[0]),int(x[1]))).groupByKey().map(lambda x: (x[0],list(set(x[1])))).collect()

usersDict = {user[0]:user[1] for user in users}

userCombs = list(combinations(range(1,len(usersDict.keys())+1),2))

userCombs = sc.parallelize(userCombs)

def getEdges(pair):
    user1 = usersDict[pair[0]]
    user2 = usersDict[pair[1]]
    return (pair[0],pair[1],len(np.intersect1d(user1,user2)))

edges = userCombs.map(getEdges).filter(lambda (x,y,z): z >= 9).map(lambda (x,y,z): (x,y)).collect()

G = nx.Graph()
G.add_edges_from(edges)

m = nx.number_of_edges(G)
nodeDegree = {}
for node in G.nodes():
    nodeDegree[node] = G.degree(node)

betw = nx.edge_betweenness_centrality(G,normalized=False)

for k,v in betw.iteritems():
    betw[k] = np.round(v,3)

betweenness = sc.parallelize(betw.items())

betweenness = betweenness.map(lambda ((x,y),z): (z,(x,y))).groupByKey().map(lambda (x,y): (x,list(y))).sortByKey(False).map(lambda (x,y): y).collect()

def getModularity():
    Q = 0.0
    S = nx.connected_components(G)
    for s in S:
        edgesIJ = list(combinations(s,2))
        for edge in edgesIJ:
            Ki = nodeDegree[edge[0]]
            Kj = nodeDegree[edge[1]]
            try:
                val = betw[edge]
                Aij = 1.0
                Q += (Aij - (Ki * Kj) / (2.0 * float(m)))
            except KeyError:
                Aij = 0.0
                Q += (Aij - (Ki * Kj) / (2.0 * float(m)))
    return(Q/(2.0*m))

l1 = []
l2 = []
Q = getModularity()
components = list(nx.connected_components(G))
for i,edgesToRemove in enumerate(betweenness):
    numComp = nx.number_connected_components(G)
    G.remove_edges_from(edgesToRemove)
    if nx.number_connected_components(G) > numComp:
        l1.append(Q)
        l2.append(components)
        Q = getModularity()
        components = list(nx.connected_components(G))

del l1[0]
del l2[0]

communities = l2[np.argmax(l1)]
print(len(communities))

with open(outputFile,'wb') as fileWrite:
    for row in communities:
        a = str(sorted(list(row)))
        fileWrite.write(a)
        fileWrite.write('\n')
fileWrite.close()

end = time.time()
print("Code took {} seconds".format(np.round(end-start,2)))