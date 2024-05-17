import numpy as np
import sklearn
import itertools
from util import Graph, Edge


# taken from https://github.com/Behrouz-Babaki/COP-Kmeans/blob/master/copkmeans/cop_kmeans.py
def transitive_closure(ml, cl, n):
    ml_graph = dict()
    cl_graph = dict()
    for i in range(n):
        ml_graph[i] = set()
        cl_graph[i] = set()

    def add_both(d, i, j):
        d[i].add(j)
        d[j].add(i)

    for (i, j) in ml:
        add_both(ml_graph, i, j)

    def dfs(i, graph, visited, component):
        visited[i] = True
        for j in graph[i]:
            if not visited[j]:
                dfs(j, graph, visited, component)
        component.append(i)

    visited = [False] * n
    for i in range(n):
        if not visited[i]:
            component = []
            dfs(i, ml_graph, visited, component)
            for x1 in component:
                for x2 in component:
                    if x1 != x2:
                        ml_graph[x1].add(x2)
    for (i, j) in cl:
        add_both(cl_graph, i, j)
        for y in ml_graph[j]:
            add_both(cl_graph, i, y)
        for x in ml_graph[i]:
            add_both(cl_graph, x, j)
            for y in ml_graph[j]:
                add_both(cl_graph, x, y)

    ml_graph = {k:v for k,v in ml_graph.items() if len(v)>0}
    cl_graph = {k:v for k,v in cl_graph.items() if len(v)>0}

    return ml_graph, cl_graph


def graph_from_knn(ind_knn, weight_knn):
    n_instance = ind_knn.shape[0]
    graph = Graph(n_instance)
    for i in range(ind_knn.shape[0]):
        for index_nn in range(ind_knn.shape[1]):
            j = ind_knn[i,index_nn]
            if i == j:
                continue
            weight_ij = weight_knn[i,index_nn]
            edge1 = Edge(i,j,weight_ij)
            edge2 = Edge(j,i,weight_ij)
            if not edge1 in graph.adj[i]:
                graph.adj[i].add(edge1)
                graph.adj[j].add(edge2)
                graph.node_degrees[i] += weight_ij
                graph.node_degrees[j] += weight_ij
                graph.sum_degrees += 2*weight_ij
    return graph

def generate_constraints_pairwise_fast(y, N_ML, N_CL, X, args, weight_knn):
    n_instance = y.shape[0]
    mls = []
    cls = []
    while len(mls) < N_ML:
        # print(len(mls))
        while True:
            i, j = np.random.randint(0, n_instance, size=2)
            if i != j:
                break
        if y[i] == y[j]:
            mls.append([i,j])
    while len(cls) < N_CL:
        while True:
            i, j = np.random.randint(0, n_instance, size=2)
            if i != j:
                break
        if y[i] != y[j]:
            cls.append([i,j])

    if n_instance <= 2.5 * args.sampling_size:
        cls = np.array(cls, dtype=int).reshape((-1, 2))
        mls = np.array(mls, dtype=int).reshape((-1, 2))
        ml_graph, cl_graph = transitive_closure(mls, cls, n_instance)
        mls = []
        cls = []
        for key in ml_graph.keys():
            mls.extend([[key, value] for value in ml_graph[key]])
        for key in cl_graph.keys():
            cls.extend([[key, value] for value in cl_graph[key]])
    N_ML = len(mls)
    N_CL = len(cls)
    mls, cls = np.array(mls), np.array(cls)
    if args.metric == 'euclidean':
        weight_max, weight_min = 1, 0
        dists_ml = np.sum((X[mls[:,0]] - X[mls[:,1]])**2, axis=-1)
        weights_ml = np.exp(-dists_ml / (2*args.sigmasq))
        weights_ml = (weight_max - weights_ml) * args.constraint_weight
        dists_cl = np.sum((X[cls[:, 0]] - X[cls[:, 1]]) ** 2, axis=-1)
        weights_cl = np.exp(-dists_cl / (2 * args.sigmasq))
        weights_cl = ((weight_min - weights_cl) * N_ML/N_CL) * args.constraint_weight
    elif args.metric == 'cosine':
        if args.normalization == "MinMaxScaler":
            weight_max, weight_min = 1, 0
        elif args.normalization == "StandardScaler":
            weight_max, weight_min = 1, -1
        else:
            raise Exception("not implemented")
        weights_ml = np.sum(X[mls[:,0]] * X[mls[:,1]], axis=1) / (np.linalg.norm(X[mls[:,0]], axis=1) * np.linalg.norm(X[mls[:,1]], axis=1))
        weights_ml = (weight_max - weights_ml) * args.constraint_weight
        weights_cl = np.sum(X[cls[:,0]] * X[cls[:,1]], axis=1) / (np.linalg.norm(X[cls[:,0]], axis=1) * np.linalg.norm(X[cls[:,1]], axis=1))
        weights_cl = ((weight_min - weights_cl) * N_ML / N_CL) * args.constraint_weight
    else:
        raise Exception("not implemented")
    graph_con = Graph(n_instance)
    for index in range(mls.shape[0]):
        i,j = mls[index]
        if i==j:
            continue
        weight_ij = weights_ml[index]
        edge1 = Edge(i, j, weight_ij)
        edge2 = Edge(j, i, weight_ij)
        if not edge1 in graph_con.adj[i]:
            graph_con.adj[i].add(edge1)
            graph_con.adj[j].add(edge2)
            graph_con.node_degrees[i] += weight_ij
            graph_con.node_degrees[j] += weight_ij
            graph_con.sum_degrees += 2 * weight_ij
    for index in range(cls.shape[0]):
        i, j = cls[index]
        weight_ij = weights_cl[index]
        edge1 = Edge(i, j, weight_ij)
        edge2 = Edge(j, i, weight_ij)
        if not edge1 in graph_con.adj[i]:
            graph_con.adj[i].add(edge1)
            graph_con.adj[j].add(edge2)
            graph_con.node_degrees[i] += weight_ij
            graph_con.node_degrees[j] += weight_ij
            graph_con.sum_degrees += 2 * weight_ij
    return graph_con


def generate_constraints_label_fast(y, N_PL, N_NL, X_fea, args, weight_knn):
    n_instance = y.shape[0]
    R = y.shape[0]
    Label = np.unique(y)
    k = len(Label)
    PL = np.zeros([R, k])
    NL = np.zeros([R, k])
    t = 0
    t1 = 0
    while t<N_PL:
        X = np.random.randint(R)
        Y = np.random.randint(k)
        if PL[X,Y]==0 and y[X]==Label[Y]:
            PL[X,Y] = 1
            t = t+1
    Z = np.sum(PL, axis=-1) > 0
    while t1<N_NL:
        X = np.random.randint(R)
        Y = np.random.randint(k)
        if Z[X]==0 and NL[X,Y]==0 and y[X]!=Label[Y]:
            NL[X,Y] = -1
            t1 = t1+1

    D = -np.sum(NL,axis=-1)
    F = np.where(D==k-1)
    PL[F,:] = (NL[F,:]==0).astype(float)

    mls = []
    cls = []

    if n_instance <= 2.5 * args.sampling_size:
        for i in range(k):
            indices_pos = np.argwhere(PL[:, i] >= 0.99).flatten()
            ml_same_pos = list(itertools.combinations(indices_pos, 2))
            mls.extend(ml_same_pos)

            indices_neg = np.argwhere(NL[:, i] <= -0.99).flatten()
            cl_same_posneg = list(itertools.product(indices_pos, indices_neg))
            cls.extend(cl_same_posneg)
            # for j in range(i + 1, k):
            #     # print((PL[:,i]+PL[:,j])>0)
            #     # indices = np.concatenate(np.argwhere(PL[:,i] > 0.99).flatten(), np.argwhere(PL[:,j] > 0.99).flatten(), axis=-1)
            #     indicesi = np.argwhere(PL[:, i] > 0.99).flatten()
            #     indicesj = np.argwhere(PL[:, j] > 0.99).flatten()
            #
            #     cl_diff_pos = list(itertools.product(indicesi, indicesj))
            #     cls.extend(cl_diff_pos)

        cls = np.array(cls, dtype=int).reshape((-1, 2))
        mls = np.array(mls, dtype=int).reshape((-1, 2))
        ml_graph, cl_graph = transitive_closure(mls, cls, n_instance)
        mls = []
        cls = []
        for key in ml_graph.keys():
            mls.extend([[key, value] for value in ml_graph[key]])
        for key in cl_graph.keys():
            cls.extend([[key, value] for value in cl_graph[key]])
        for i in range(k):
            indices_pos = np.argwhere(PL[:, i] >= 0.99).flatten()
            ml_same_pos = list(itertools.combinations(indices_pos, 2))
            mls.extend(ml_same_pos)

            indices_neg = np.argwhere(NL[:, i] <= -0.99).flatten()
            cl_same_posneg = list(itertools.product(indices_pos, indices_neg))
            cls.extend(cl_same_posneg)
        k_con = args.k_con
        ml_dict = {}
        for ml in mls:
            i, j = ml
            if i not in ml_dict.keys():
                ml_dict[i] = set()
            if j not in ml_dict.keys():
                ml_dict[j] = set()
            ml_dict[i].add(j)
            ml_dict[j].add(i)
        mls = []
        for i in ml_dict.keys():
            ml_i = np.array(list(ml_dict[i]))
            np.random.shuffle(ml_i)
            ml_i = ml_i[:k_con]
            # print(ml_i)
            for j in ml_i:
                mls.append([i, j])
        cl_dict = {}
        for cl in cls:
            i, j = cl
            if i not in cl_dict.keys():
                cl_dict[i] = set()
            if j not in cl_dict.keys():
                cl_dict[j] = set()
            cl_dict[i].add(j)
            cl_dict[j].add(i)
        cls = []
        for i in cl_dict.keys():
            cl_i = np.array(list(cl_dict[i]))
            np.random.shuffle(cl_i)
            cl_i = cl_i[:k_con]
            for j in cl_i:
                cls.append([i, j])

    else:
        indices_pos_dict = {}
        indices_neg_dict = {}
        for i in range(k):
            indices_pos = np.argwhere(PL[:, i] >= 0.99).flatten()
            indices_pos_dict[Label[i]] = indices_pos
            indices_neg = np.argwhere(NL[:, i] <= -0.99).flatten()
            indices_neg_dict[Label[i]] = indices_neg
        for i in np.argwhere(Z).flatten().tolist():
            indices_pos = indices_pos_dict[y[i]]
            if indices_pos.shape[0] > 2:
                js = np.random.choice(indices_pos, size=args.k_con)
                ml_same_pos = list(itertools.product([i], list(js)))
                mls.extend(ml_same_pos)
        for i in np.argwhere(D > 0).flatten().tolist():
            labels_i = np.argwhere(NL[i,:] <= -0.99).flatten().tolist()
            labels_i = [Label[x] for x in labels_i]
            indices_pos_all = []
            for label_i in labels_i:
                if label_i in indices_pos_dict.keys():
                    indices_pos_all.append(indices_pos_dict[label_i])
            if len(indices_pos_all) > 0:
                indices_pos_all = np.concatenate(indices_pos_all, axis=0)
                if indices_pos_all.shape[0] > 0:
                    js = np.random.choice(indices_pos_all, size=args.k_con)
                    cl_same_posneg = list(itertools.product([i], list(js)))
                    cls.extend(cl_same_posneg)
        for i in np.argwhere(Z).flatten().tolist():
            if y[i] in indices_neg_dict.keys():
                indices_neg = indices_neg_dict[y[i]]
                if indices_neg.shape[0] > 1:
                    js = np.random.choice(indices_neg, size=args.k_con)
                    cl_same_posneg = list(itertools.product([i], list(js)))
                    cls.extend(cl_same_posneg)
        mls = list(set(mls))
        cls = list(set(cls))

    mls, cls = np.array(mls), np.array(cls)
    # print("the size of mls is {} and cls is {}".format(mls.shape[0], cls.shape[0]))
    if args.metric == 'euclidean':
        weight_max, weight_min = 1, 0
        dists_ml = np.sum((X_fea[mls[:,0]] - X_fea[mls[:,1]])**2, axis=-1)
        weights_ml = np.exp(-dists_ml / (2*args.sigmasq))
        weights_ml = (weight_max - weights_ml) * args.constraint_weight
        dists_cl = np.sum((X_fea[cls[:, 0]] - X_fea[cls[:, 1]]) ** 2, axis=-1)
        weights_cl = np.exp(-dists_cl / (2 * args.sigmasq))
        weights_cl = ((weight_min - weights_cl) * mls.shape[0]/cls.shape[0]) * args.constraint_weight
    elif args.metric == 'cosine':
        if args.normalization == "MinMaxScaler":
            weight_max, weight_min = 1, 0
        elif args.normalization == "StandardScaler":
            weight_max, weight_min = 1, -1
        else:
            raise Exception("not implemented")
        weights_ml = np.sum(X_fea[mls[:, 0]] * X_fea[mls[:, 1]], axis=1) / (
                    np.linalg.norm(X_fea[mls[:, 0]], axis=1) * np.linalg.norm(X_fea[mls[:, 1]], axis=1))
        weights_ml = (weight_max - weights_ml) * args.constraint_weight
        weights_cl = np.sum(X_fea[cls[:, 0]] * X_fea[cls[:, 1]], axis=1) / (
                    np.linalg.norm(X_fea[cls[:, 0]], axis=1) * np.linalg.norm(X_fea[cls[:, 1]], axis=1))
        weights_cl = ((weight_min - weights_cl) * mls.shape[0] / cls.shape[0]) * args.constraint_weight
    else:
        raise Exception("not implemented")

    graph_con = Graph(n_instance)
    for index in range(mls.shape[0]):
        i,j = mls[index]
        if i == j:
            continue
        weight_ij = weights_ml[index]
        edge1 = Edge(i, j, weight_ij)
        edge2 = Edge(j, i, weight_ij)
        if not edge1 in graph_con.adj[i]:
            graph_con.adj[i].add(edge1)
            graph_con.adj[j].add(edge2)
            graph_con.node_degrees[i] += weight_ij
            graph_con.node_degrees[j] += weight_ij
            graph_con.sum_degrees += 2 * weight_ij
    for index in range(cls.shape[0]):
        i, j = cls[index]
        weight_ij = weights_cl[index]
        edge1 = Edge(i, j, weight_ij)
        edge2 = Edge(j, i, weight_ij)
        if not edge1 in graph_con.adj[i]:
            graph_con.adj[i].add(edge1)
            graph_con.adj[j].add(edge2)
            graph_con.node_degrees[i] += weight_ij
            graph_con.node_degrees[j] += weight_ij
            graph_con.sum_degrees += 2 * weight_ij
    return graph_con


def knn_k_estimating(n_cluster, n_instance, knn_constant):
    knn_k = int(np.ceil(knn_constant * (n_instance/n_cluster) / (np.log2(n_instance) ** 2)))
    return knn_k


