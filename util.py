import numpy as np
from sklearn import metrics
import itertools
from tqdm import tqdm

class Graph():
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.adj = dict()
        self.node_degrees = dict()
        self.sum_degrees = 0
        for i in range(self.num_nodes):
            self.adj[i] = set()
            self.node_degrees[i] = 0

    def get_subgraph(self, ind):
        graph = Graph(ind.shape[0])
        ind_reverse = {ind[i]:i for i in range(ind.shape[0])}
        for i in ind:
            for edge in self.adj[i]:
                if edge.j in ind:
                    edge1 = Edge(ind_reverse[edge.i], ind_reverse[edge.j], edge.weight)
                    graph.adj[ind_reverse[edge.i]].add(edge1)
                    graph.node_degrees[ind_reverse[edge.i]] += edge.weight
                    graph.sum_degrees += edge.weight
        return graph

    def to_affinity(self):
        A = np.zeros([self.num_nodes, self.num_nodes])
        for i in self.adj.keys():
            for edge in self.adj[i]:
                A[edge.i, edge.j] = edge.weight
        return A


class Edge():
    def __init__(self, i, j, weight):
        self.i = i
        self.j = j
        self.weight = weight

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self.i != other.i:
                return False
            elif self.j != other.j:
                return False
            elif self.weight != other.weight:
                return False
            else:
                return True
        else:
            return False

    def __hash__(self):
        return hash((self.i,self.j,self.weight))

def get_graph(A):
    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]
    num_nodes = A.shape[0]
    graph = Graph(num_nodes)
    for i in range(A.shape[0]):
        for j in range(i+1, A.shape[1]):
            if A[i,j] != A[j,i]:
                print("A[i,j] != A[j,i]")
            weight = (A[i,j]+A[j,i])/2
            if weight == 0:
                continue
            edge1 = Edge(i,j,weight)
            edge2 = Edge(j,i,weight)
            if not edge1 in graph.adj[i]:
                graph.adj[i].add(edge1)
                graph.adj[j].add(edge2)
                graph.node_degrees[i] += weight
                graph.node_degrees[j] += weight
                graph.sum_degrees += 2*weight
    return graph

def dendrogram_purity(t, y):
    y = y.astype(int)
    y_onehot = np.zeros((y.shape[0], y.max()+1))
    y_onehot[np.arange(y.shape[0]), y] = 1
    cluster_dict = {}
    pairs = []
    for i in range(y_onehot.shape[1]):
        indicesi = np.argwhere(y_onehot[:,i] == 1).flatten()
        pairs_i = list(itertools.permutations(indicesi, 2))
        pairs.append(pairs_i)
        cluster_indices = np.argwhere(y_onehot[:,i] == 1).flatten().tolist()
        for j in indicesi:
            cluster_dict[j] = cluster_indices
    purity_list = []
    for index, pairs_i in enumerate(tqdm(pairs)):
        for pair in pairs_i:
            i,j = pair
            nodei = t.search_nodes(name=i)
            nodej = t.search_nodes(name=j)
            ancestor = t.get_common_ancestor(nodei[0],nodej[0])
            ancestor_leaves = [int(i.name) for i in ancestor.get_leaves()]
            cluster_indices = cluster_dict[i]
            purity = len(set(ancestor_leaves).intersection(cluster_indices)) / len(ancestor_leaves)
            purity_list.append(purity)
    return np.mean(purity_list)


def dendrogram_purity_expected(t, y, n_sample=1000):
    n_instance = y.shape[0]
    y = y.astype(int)
    y_onehot = np.zeros((y.shape[0], y.max() + 1))
    y_onehot[np.arange(y.shape[0]), y] = 1
    cluster_dict = {}
    for i in range(y_onehot.shape[1]):
        indicesi = np.argwhere(y_onehot[:, i] == 1).flatten()
        cluster_indices = np.argwhere(y_onehot[:, i] == 1).flatten().tolist()
        for j in indicesi:
            cluster_dict[j] = cluster_indices
    purity_list = []
    leaves_dict = {}
    for leaf in t.get_leaves():
        leaves_dict[int(leaf.name)] = leaf
    for index_sample in tqdm(range(n_sample)):
        nodeID_i = np.random.randint(n_instance)
        cluster_indices = cluster_dict[nodeID_i]
        nodeID_j = np.random.choice(cluster_indices)
        nodei = leaves_dict[nodeID_i]
        nodej = leaves_dict[nodeID_j]
        ancestor = t.get_common_ancestor(nodei, nodej)
        ancestor_leaves = [int(i.name) for i in ancestor.get_leaves()]
        purity = len(set(ancestor_leaves).intersection(cluster_indices)) / len(ancestor_leaves)
        purity_list.append(purity)
    return np.mean(purity_list)