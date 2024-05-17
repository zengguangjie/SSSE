from multiprocessing import get_context

import numba as nb
import numpy as np
from partitioning_single_graph import FlatSSE
from util import Graph, Edge
from queue import Queue
import scipy
from concurrent import futures


def partitioning_single_thread(ind, subgraph, subgraph_con, args):
    if args.mergestop_SE == 0:
        flatSSE = FlatSSE(subgraph, subgraph_con, None, False, False)
    else:
        flatSSE = FlatSSE(subgraph, subgraph_con, None, False, True)
    sub_y_pred = flatSSE.build_tree()
    return ind, sub_y_pred

class FlatSSSE():
    def __init__(self, graph, graph_con, n_cluster, sampling_size, args):
        self.graph = graph
        self.graph_con = graph_con
        self.n_cluster = n_cluster
        self.sampling_size = sampling_size
        self.args = args
        self.n_instance = graph.num_nodes
        self.y_pred_list = [] # clustering of each round.


    def sampling_random_multi(self):
        n_instance = self.graph.num_nodes
        sampled_set_list = []
        subgraph_list = []
        subgraph_con_list = []
        visited = np.zeros([n_instance], dtype=bool)
        while n_instance - np.sum(visited) >= 2 * self.sampling_size:
            ind = np.array(np.random.choice(np.argwhere(np.logical_not(visited)).flatten(), size=self.sampling_size, replace=False))
            visited[ind] = True
            subgraph = self.graph.get_subgraph(ind)
            subgraph_con = self.graph_con.get_subgraph(ind)
            sampled_set_list.append(ind)
            subgraph_list.append(subgraph)
            subgraph_con_list.append(subgraph_con)
        ind = np.argwhere(np.logical_not(visited)).flatten()
        subgraph = self.graph.get_subgraph(ind)
        subgraph_con = self.graph_con.get_subgraph(ind)
        sampled_set_list.append(ind)
        subgraph_list.append(subgraph)
        subgraph_con_list.append(subgraph_con)
        return sampled_set_list, subgraph_list, subgraph_con_list

    def sampling_neighbor_multi(self):
        n_instance = self.graph.num_nodes
        sampled_set_list = []
        subgraph_list = []
        subgraph_con_list = []
        visited = np.zeros([n_instance], dtype=bool)
        while n_instance - np.sum(visited) >= 2*self.sampling_size:
            sampled_set = set()
            while len(sampled_set) < self.sampling_size:
                # print(np.argwhere(np.logical_not(visited)))
                seeds = np.random.choice(np.argwhere(np.logical_not(visited)).flatten(), size=self.args.n_seeds, replace=False)
                q = Queue()
                for seed in seeds:
                    q.put(seed)
                while not q.empty():
                    vertexID = q.get()
                    if visited[vertexID]:
                        continue
                    # assert visited[vertexID] == False
                    # print(vertexID, len(sampled_set))
                    visited[vertexID] = True
                    sampled_set.add(vertexID)
                    if len(sampled_set) >= self.sampling_size:
                        ind = np.array(list(sampled_set))
                        subgraph = self.graph.get_subgraph(ind)
                        subgraph_con = self.graph_con.get_subgraph(ind)
                        sampled_set_list.append(ind)
                        subgraph_list.append(subgraph)
                        subgraph_con_list.append(subgraph_con)
                        break
                    for edge in self.graph.adj[vertexID]:
                        if not visited[edge.j]:
                            q.put(edge.j)



        ind = np.argwhere(np.logical_not(visited)).flatten()
        subgraph = self.graph.get_subgraph(ind)
        subgraph_con = self.graph_con.get_subgraph(ind)
        sampled_set_list.append(ind)
        subgraph_list.append(subgraph)
        subgraph_con_list.append(subgraph_con)

        return sampled_set_list, subgraph_list, subgraph_con_list


    def run_parallel(self, mustlink_first=False):
        clusters = {i:{i} for i in range(self.n_instance)}
        while self.graph.num_nodes > 2.5*self.sampling_size:
            last_clusters = clusters
            if self.args.sampling == 'neighbor':
                sampled_set_list, subgraph_list, subgraph_con_list = self.sampling_neighbor_multi()
            elif self.args.sampling == 'random':
                sampled_set_list, subgraph_list, subgraph_con_list = self.sampling_random_multi()
            else:
                raise Exception("not implemented")
            cur_partitioning = {}
            mp_context = get_context('spawn')
            with futures.ProcessPoolExecutor(self.args.n_threads, mp_context=mp_context) as executor:
                to_do = []
                for ind, subgraph, subgraph_con in zip(sampled_set_list, subgraph_list, subgraph_con_list):
                    job = executor.submit(partitioning_single_thread, ind, subgraph, subgraph_con, self.args)
                    to_do.append(job)
                for future in futures.as_completed(to_do):
                    ind, sub_y_pred = future.result()
                    # print(ind)
                    if len(cur_partitioning.keys()) == 0:
                        cur_index = 0
                    else:
                        cur_index = np.max(list(cur_partitioning.keys())) + 1
                    sub_partitioning = {}
                    for index, y_i in enumerate(np.unique(sub_y_pred)):
                        sub_partitioning[index] = np.argwhere(sub_y_pred == y_i).flatten()
                    for index in sub_partitioning.keys():
                        cur_partitioning[cur_index + index] = [ind[valuei] for valuei in sub_partitioning[index]]

            vertices_transform_dict_reverse = {}  # old vertices -> new vertices.
            for i_new in cur_partitioning.keys():
                for i_old in cur_partitioning[i_new]:
                    vertices_transform_dict_reverse[i_old] = i_new

            # clusters update
            clusters = {}
            for index in cur_partitioning.keys():
                clusters[index] = set()
                for value in cur_partitioning[index]:
                    clusters[index].update(last_clusters[value])
            # graph update no change
            graph = Graph(len(cur_partitioning.keys()))
            adj_dict = {}
            for i_cur in range(self.graph.num_nodes):
                for edge in self.graph.adj[i_cur]:
                    j_cur = edge.j
                    i_new, j_new = vertices_transform_dict_reverse[i_cur], vertices_transform_dict_reverse[j_cur]
                    if tuple([i_new, j_new]) in adj_dict.keys():
                        adj_dict[tuple([i_new, j_new])] += edge.weight
                    else:
                        adj_dict[tuple([i_new, j_new])] = edge.weight
            for key in adj_dict.keys():
                i_new, j_new = key
                if i_new == j_new:
                    continue
                edge1 = Edge(i_new,j_new,adj_dict[key])
                edge2 = Edge(j_new,i_new,adj_dict[key])
                if not edge1 in graph.adj[i_new]:
                    graph.adj[i_new].add(edge1)
                    graph.node_degrees[i_new] += edge1.weight
                    graph.adj[j_new].add(edge2)
                    graph.node_degrees[j_new] += edge2.weight
                    graph.sum_degrees += 2*edge1.weight
            self.graph = graph
            graph_con = Graph(len(cur_partitioning.keys()))
            adj_con_dict = {}
            for i_cur in range(self.graph_con.num_nodes):
                for edge in self.graph_con.adj[i_cur]:
                    j_cur = edge.j
                    i_new, j_new = vertices_transform_dict_reverse[i_cur], vertices_transform_dict_reverse[j_cur]
                    if tuple([i_new,j_new]) in adj_con_dict.keys():
                        adj_con_dict[tuple([i_new,j_new])] += edge.weight
                    else:
                        adj_con_dict[tuple([i_new,j_new])] = edge.weight
            for key in adj_con_dict.keys():
                i_new, j_new = key
                if i_new == j_new:
                    continue
                edge1 = Edge(i_new,j_new,adj_con_dict[key])
                edge2 = Edge(j_new,i_new,adj_con_dict[key])
                if not edge1 in graph_con.adj[i_new]:
                    graph_con.adj[i_new].add(edge1)
                    graph_con.node_degrees[i_new] += edge1.weight
                    graph_con.adj[j_new].add(edge2)
                    graph_con.node_degrees[j_new] += edge2.weight
                    graph_con.sum_degrees += 2*edge1.weight
            self.graph_con = graph_con

        # clustering on remaining graphs
        if self.args.mergestop_SE == 0:
            flatSSE = FlatSSE(self.graph, self.graph_con, self.n_cluster, mustlink_first=mustlink_first, mergestop_SE=False)
        else:
            flatSSE = FlatSSE(self.graph, self.graph_con, self.n_cluster, mustlink_first=mustlink_first, mergestop_SE=True)
        sub_y_pred = flatSSE.build_tree()
        y_pred = np.zeros([self.n_instance])
        for index, value in enumerate(sub_y_pred):
            y_pred[np.array(list(clusters[index])).astype(int)] = value
        # transform the cluster_index to be 0~m
        _, indices_inverse = np.unique(y_pred, return_inverse=True)
        y_pred = indices_inverse
        self.y_pred_list.append(y_pred)
        return y_pred








