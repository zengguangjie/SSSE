import numpy as np
from util import Graph, Edge
from queue import Queue
from concurrent import futures
from hierarchical_single_graph import PartitionTree_SSE
from ete3 import Tree

def hierarchical_single_thread(ind, subgraph, subgraph_con):
    partitionTree_SSSE = PartitionTree_SSE(subgraph, subgraph_con, False)
    root_id, hierarchical_tree_node, sub_y_pred, cluster_nodeIDs = partitionTree_SSSE.build_tree(k=2)
    return ind, root_id, hierarchical_tree_node, sub_y_pred, cluster_nodeIDs


class TreeSSSE():
    def __init__(self, graph, graph_con, height, sampling_size, args):
        self.graph = graph
        self.graph_con = graph_con
        self.height = height
        self.sampling_size = sampling_size
        self.n_instance = graph.num_nodes
        self.args = args

    def sampling_neighbor_multi(self):
        n_instance = self.graph.num_nodes
        sampled_set_list = []
        subgraph_list = []
        subgraph_con_list = []
        visited = np.zeros([n_instance], dtype=bool)
        while n_instance - np.sum(visited) >= 2 * self.sampling_size:
            sampled_set = set()
            while len(sampled_set) < self.sampling_size:
                seeds = np.random.choice(np.argwhere(np.logical_not(visited)).flatten(), size=self.args.n_seeds,
                                         replace=False)
                q = Queue()
                for seed in seeds:
                    q.put(seed)
                while not q.empty():
                    vertexID = q.get()
                    if visited[vertexID]:
                        continue
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
                            # sampled_set.add(edge.j)

        ind = np.argwhere(np.logical_not(visited)).flatten()
        subgraph = self.graph.get_subgraph(ind)
        subgraph_con = self.graph_con.get_subgraph(ind)
        sampled_set_list.append(ind)
        subgraph_list.append(subgraph)
        subgraph_con_list.append(subgraph_con)

        return sampled_set_list, subgraph_list, subgraph_con_list

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


    def run_parallel(self):
        subtrees = {}
        for i in range(self.graph.num_nodes):
            subtrees[i] = Tree(name=i)
        while self.graph.num_nodes > 2 * self.sampling_size:
            last_subtrees = subtrees
            if self.args.sampling == 'neighbor':
                sampled_set_list, subgraph_list, subgraph_con_list = self.sampling_neighbor_multi()
            elif self.args.sampling == 'random':
                sampled_set_list, subgraph_list, subgraph_con_list = self.sampling_random_multi()
            else:
                raise Exception("not implemented")
            cur_trees = {}
            cur_partitioning = {}
            with futures.ProcessPoolExecutor(self.args.n_threads) as executor:
                to_do = []
                for ind, subgraph, subgraph_con in zip(sampled_set_list, subgraph_list, subgraph_con_list):
                    job = executor.submit(hierarchical_single_thread, ind, subgraph, subgraph_con)
                    to_do.append(job)
                for future in futures.as_completed(to_do):
                    ind, root_id, hierarchical_tree_node, sub_y_pred, cluster_nodeIDs = future.result()
                    if len(cur_partitioning.keys()) == 0:
                        cur_index = 0
                    else:
                        cur_index = np.max(list(cur_partitioning.keys())) + 1
                    sub_partitioning = {}
                    for index, cluster_nodeID in enumerate(cluster_nodeIDs):
                        sub_partitioning[index] = np.argwhere(sub_y_pred == cluster_nodeID).flatten()
                        if hierarchical_tree_node[cluster_nodeID].children is not None:
                            t_ete = Tree(name=str(cluster_nodeID))
                        else:
                            t_ete = Tree(name=ind[int(cluster_nodeID)])
                            cur_trees[cur_index + index] = t_ete
                            continue
                        queue = Queue()
                        queue.put(t_ete)
                        while not queue.empty():
                            t_ete_node = queue.get()
                            nodeID = int(t_ete_node.name)
                            tree_node = hierarchical_tree_node[nodeID]
                            if tree_node.children is not None:
                                for child in tree_node.children:
                                    if hierarchical_tree_node[child].children is not None:
                                        t_ete_child = t_ete_node.add_child(name=str(child))
                                        queue.put(t_ete_child)
                                    else:
                                        t_ete_child = t_ete_node.add_child(name=ind[child])
                        cur_trees[cur_index + index] = t_ete

                    for index in sub_partitioning.keys():
                        cur_partitioning[cur_index + index] = [ind[valuei] for valuei in sub_partitioning[index]]

            vertices_transform_dict_reverse = {}  # old vertices -> new vertices.
            for i_new in cur_partitioning.keys():
                for i_old in cur_partitioning[i_new]:
                    vertices_transform_dict_reverse[i_old] = i_new

            # subtrees update
            subtrees = {}
            for index in cur_trees.keys():
                subtrees[index] = cur_trees[index]
                for leaf in cur_trees[index].get_leaves():
                    leaf.add_child(last_subtrees[int(leaf.name)])

            # graph update
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
                edge1 = Edge(i_new, j_new, adj_dict[key])
                edge2 = Edge(j_new, i_new, adj_dict[key])
                if not edge1 in graph.adj[i_new]:
                    graph.adj[i_new].add(edge1)
                    graph.node_degrees[i_new] += edge1.weight
                    graph.adj[j_new].add(edge2)
                    graph.node_degrees[j_new] += edge2.weight
                    graph.sum_degrees += 2 * edge1.weight
            self.graph = graph
            graph_con = Graph(len(cur_partitioning.keys()))
            adj_con_dict = {}
            for i_cur in range(self.graph_con.num_nodes):
                for edge in self.graph_con.adj[i_cur]:
                    j_cur = edge.j
                    i_new, j_new = vertices_transform_dict_reverse[i_cur], vertices_transform_dict_reverse[j_cur]
                    if tuple([i_new, j_new]) in adj_con_dict.keys():
                        adj_con_dict[tuple([i_new, j_new])] += edge.weight
                    else:
                        adj_con_dict[tuple([i_new, j_new])] = edge.weight
            for key in adj_con_dict.keys():
                i_new, j_new = key
                if i_new == j_new:
                    continue
                edge1 = Edge(i_new, j_new, adj_con_dict[key])
                edge2 = Edge(j_new, i_new, adj_con_dict[key])
                if not edge1 in graph_con.adj[i_new]:
                    graph_con.adj[i_new].add(edge1)
                    graph_con.node_degrees[i_new] += edge1.weight
                    graph_con.adj[j_new].add(edge2)
                    graph_con.node_degrees[j_new] += edge2.weight
                    graph_con.sum_degrees += 2 * edge1.weight
            self.graph_con = graph_con

        # clustering on remaining graphs
        partitionTree_SSSE = PartitionTree_SSE(self.graph, self.graph_con, False)
        root_id, hierarchical_tree_node = partitionTree_SSSE.build_tree(k=None)
        t_ete = Tree(name=str(root_id))
        queue = Queue()
        queue.put(t_ete)
        while not queue.empty():
            t_ete_node = queue.get()
            nodeID = int(t_ete_node.name)
            tree_node = hierarchical_tree_node[nodeID]
            if tree_node.children is not None:
                for child in tree_node.children:
                    if hierarchical_tree_node[child].children is not None:
                        t_ete_child = t_ete_node.add_child(name=str(child))
                        queue.put(t_ete_child)
                    else:
                        t_ete_child = t_ete_node.add_child(subtrees[child])
        return t_ete







