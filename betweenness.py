import sys
import numpy as np
import networkx as nx
import time
from itertools import combinations
from networkx import *
from pyspark import SparkContext

sc = SparkContext(appName="Betweenness")

start = time.time()

inputFile = sys.argv[1]
outputFile = 'Nikhit_Mago_Betweenness.txt'

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

betw = nx.edge_betweenness_centrality(G,normalized=False)
print(len(betw))

b = sorted(betw.items(),key = lambda x: x[0])
with open(outputFile,'wb') as fileWrite:
    for row in b:
        a = '({},{},{})'.format(row[0][0],row[0][1],row[1])
        fileWrite.write(a)
        fileWrite.write('\n')
fileWrite.close()

end = time.time()
print("Code took {} seconds".format(np.round(end-start,2)))