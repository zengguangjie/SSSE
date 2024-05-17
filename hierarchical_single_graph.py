import copy
import math
import heapq
import numba as nb
import numpy as np
from queue import Queue
from ete3 import Tree

def get_id():
    i = 0
    while True:
        yield i
        i += 1

def graph_parse(adj_matrix):
    g_num_nodes = adj_matrix.shape[0]
    adj_table = {}
    VOL = 0
    node_vol = []
    for i in range(g_num_nodes):
        n_v = 0
        adj = set()
        for j in range(g_num_nodes):
            if adj_matrix[i,j] != 0:
                n_v += adj_matrix[i,j]
                VOL += adj_matrix[i,j]
                adj.add(j)
        adj_table[i] = adj
        node_vol.append(n_v)
    return g_num_nodes,VOL,node_vol,adj_table

@nb.jit(nopython=True)
def cut_volume(adj_matrix,p1,p2):
    c12 = 0
    for i in range(len(p1)):
        for j in range(len(p2)):
            c = adj_matrix[p1[i],p2[j]]
            if c != 0:
                c12 += c
    return c12


def merge(new_ID, id1, id2, cut_v, cut_v_con, node_dict):
    new_partition = node_dict[id1].partition + node_dict[id2].partition
    v = node_dict[id1].vol + node_dict[id2].vol
    g = node_dict[id1].g + node_dict[id2].g - 2 * cut_v
    v_con = node_dict[id1].vol + node_dict[id2].vol
    g_con = node_dict[id1].g_con + node_dict[id2].g_con - 2*cut_v_con
    child_h = max(node_dict[id1].child_h,node_dict[id2].child_h) + 1
    new_node = PartitionTreeNode(ID=new_ID,partition=new_partition,children={id1,id2},
                                 g=g, vol=v, g_con=g_con, vol_con=v_con,child_h= child_h,child_cut = cut_v)
    node_dict[id1].parent = new_ID
    node_dict[id2].parent = new_ID
    node_dict[new_ID] = new_node

def compressNode(node_dict, node_id, parent_id):
    p_child_h = node_dict[parent_id].child_h
    node_children = node_dict[node_id].children
    node_dict[parent_id].child_cut += node_dict[node_id].child_cut
    node_dict[parent_id].children.remove(node_id)
    node_dict[parent_id].children = node_dict[parent_id].children.union(node_children)
    for c in node_children:
        node_dict[c].parent = parent_id
    com_node_child_h = node_dict[node_id].child_h
    node_dict.pop(node_id)

    if (p_child_h - com_node_child_h) == 1:
        while True:
            max_child_h = max([node_dict[f_c].child_h for f_c in node_dict[parent_id].children])
            if node_dict[parent_id].child_h == (max_child_h + 1):
                break
            node_dict[parent_id].child_h = max_child_h + 1
            parent_id = node_dict[parent_id].parent
            if parent_id is None:
                break

def child_tree_deepth(node_dict,nid):
    node = node_dict[nid]
    deepth = 0
    while node.parent is not None:
        node = node_dict[node.parent]
        deepth+=1
    deepth += node_dict[nid].child_h
    return deepth



class PartitionTreeNode():
    def __init__(self, ID, partition, vol, g, vol_con, g_con, children:set = None,parent = None,child_h = 0, child_cut = 0):
        self.ID = ID
        self.partition = partition
        self.parent = parent
        self.children = children
        self.vol = vol
        self.g = g
        self.vol_con = vol_con
        self.g_con = g_con
        self.merged = False
        self.child_h = child_h
        self.child_cut = child_cut

    def __str__(self):
        return "{" + "{}:{}".format(self.__class__.__name__, self.gatherAttrs()) + "}"

    def gatherAttrs(self):
        return ",".join("{}={}"
                        .format(k, getattr(self, k))
                        for k in self.__dict__.keys())

class PartitionTree_SSE():
    def __init__(self,graph, graph_con, mustlink_first=False):
        self.mustlink_first = mustlink_first
        adj_matrix = graph.to_affinity()
        adj_matrix_con = graph_con.to_affinity()
        self.adj_matrix = adj_matrix
        self.adj_matrix_con = adj_matrix_con
        self.tree_node = {}
        self.g_num_nodes, self.VOL, self.node_vol, self.adj_table = graph_parse(adj_matrix)
        _, _, self.node_vol_con, self.adj_table_con = graph_parse(adj_matrix_con)
        self.id_g = get_id()
        self.leaves = []
        self.SE = 0
        self.SSE = 0
        self.build_leaves()
        # print(self.VOL)

    def CombineDelta(self, node1, node2, cut_v, cut_v_con, g_vol):
        v1 = node1.vol
        v2 = node2.vol
        g1 = node1.g
        g2 = node2.g
        v12 = v1 + v2
        if len(node1.partition)==1:
            cut_v_con -= node1.g_con/2
        if len(node2.partition)==1:
            cut_v_con -= node2.g_con/2
        return -(2 * (cut_v+cut_v_con) / g_vol) * np.log2(g_vol / v12)

    def CompressDelta(self, node1, p_node, g_vol):
        assert node1.children is not None
        children = node1.children
        cut_sum = 0
        for child in children:
            cut_sum += self.tree_node[child].g
            if len(self.tree_node[child].partition) == 1:
                continue
            cut_sum += self.tree_node[child].g_con
        return -((cut_sum - node1.g - node1.g_con) / g_vol) * np.log2(node1.vol / p_node.vol)

    def build_leaves(self):
        for vertex in range(self.g_num_nodes):
            ID = next(self.id_g)
            v = self.node_vol[vertex]
            v_con = self.node_vol_con[vertex]
            leaf_node = PartitionTreeNode(ID=ID, partition=[vertex], g = v, vol=v, g_con = v_con, vol_con = v_con)
            self.tree_node[ID] = leaf_node
            self.leaves.append(ID)
            # self.root_node.children.add(ID)
            if v > 0:
                self.SE -= (v/self.VOL) * np.log2(v/self.VOL)
                self.SSE -= (v/self.VOL) * np.log2(v/self.VOL)

    def entropy(self,node_dict = None):
        if node_dict is None:
            node_dict = self.tree_node
        ent = 0
        for node_id,node in node_dict.items():
            if node.parent is not None:
                node_p = node_dict[node.parent]
                node_vol = node.vol
                node_g = node.g
                node_p_vol = node_p.vol
                ent += - (node_g / self.VOL) * np.log2(node_vol / node_p_vol)
        return ent

    def __build_k_tree(self, g_vol, nodes_dict:dict, k=None,):
        min_heap = []
        cmp_heap = []
        nodes_ids = nodes_dict.keys()
        new_id = None
        for i in nodes_ids:
            for j in self.adj_table[i]:
                if j > i:
                    n1 = nodes_dict[i]
                    n2 = nodes_dict[j]
                    if len(n1.partition) == 1 and len(n2.partition) == 1:
                        cut_v = self.adj_matrix[n1.partition[0],n2.partition[0]]
                    else:
                        cut_v = cut_volume(self.adj_matrix, p1=np.array(n1.partition), p2=np.array(n2.partition))
                    if len(n1.partition) == 1 and len(n2.partition) == 1:
                        cut_v_con = self.adj_matrix_con[n1.partition[0], n2.partition[0]]
                    else:
                        cut_v_con = cut_volume(self.adj_matrix_con, p1=np.array(n1.partition), p2=np.array(n2.partition))
                    pair_mustlink = 0
                    if self.mustlink_first:
                        if cut_v_con > 0:
                            pair_mustlink = 1
                        elif cut_v_con < 0:
                            pair_mustlink = -1

                    diff = self.CombineDelta(nodes_dict[i], nodes_dict[j], cut_v, cut_v_con, g_vol)
                    heapq.heappush(min_heap, (-pair_mustlink, diff, i, j, cut_v, cut_v_con))
        unmerged_count = len(nodes_ids)
        while unmerged_count > 1:
            if len(min_heap) == 0:
                break
            pair_mustlink, diff, id1, id2, cut_v, cut_v_con = heapq.heappop(min_heap)
            pair_mustlink = -pair_mustlink
            if nodes_dict[id1].merged or nodes_dict[id2].merged:
                continue
            nodes_dict[id1].merged = True
            nodes_dict[id2].merged = True
            new_id = next(self.id_g)
            merge(new_id, id1, id2, cut_v, cut_v_con, nodes_dict)
            self.SE += diff
            self.adj_table[new_id] = self.adj_table[id1].union(self.adj_table[id2])
            for i in self.adj_table[new_id]:
                self.adj_table[i].add(new_id)
            #compress delta
            if nodes_dict[id1].child_h > 0:
                heapq.heappush(cmp_heap,[self.CompressDelta(nodes_dict[id1],nodes_dict[new_id], g_vol),id1,new_id])
            if nodes_dict[id2].child_h > 0:
                heapq.heappush(cmp_heap,[self.CompressDelta(nodes_dict[id2],nodes_dict[new_id], g_vol),id2,new_id])
            unmerged_count -= 1

            for ID in self.adj_table[new_id]:
                if not nodes_dict[ID].merged:
                    n1 = nodes_dict[ID]
                    n2 = nodes_dict[new_id]
                    cut_v = cut_volume(self.adj_matrix,np.array(n1.partition), np.array(n2.partition))
                    cut_v_con = cut_volume(self.adj_matrix_con,np.array(n1.partition), np.array(n2.partition))

                    pair_mustlink = 0
                    if self.mustlink_first:
                        if cut_v_con > 0:
                            pair_mustlink = 1
                        elif cut_v_con < 0:
                            pair_mustlink = -1
                    new_diff = self.CombineDelta(nodes_dict[ID], nodes_dict[new_id], cut_v, cut_v_con, g_vol)
                    heapq.heappush(min_heap, (-pair_mustlink, new_diff, ID, new_id, cut_v, cut_v_con))
        root = new_id

        if unmerged_count > 1:
            #combine solitary node
            # print('processing solitary node')
            assert len(min_heap) == 0
            unmerged_nodes = {i for i, j in nodes_dict.items() if not j.merged}
            new_child_h = max([nodes_dict[i].child_h for i in unmerged_nodes]) + 1

            new_id = next(self.id_g)
            new_node = PartitionTreeNode(ID=new_id,partition=list(range(self.g_num_nodes)),children=unmerged_nodes,
                                         vol=g_vol,g = 0, vol_con=None, g_con=None ,child_h=new_child_h)
            nodes_dict[new_id] = new_node

            for i in unmerged_nodes:
                nodes_dict[i].merged = True
                nodes_dict[i].parent = new_id
                if nodes_dict[i].child_h > 0:
                    heapq.heappush(cmp_heap, [self.CompressDelta(nodes_dict[i], nodes_dict[new_id], g_vol), i, new_id])
            root = new_id
        tree_node_copy = copy.deepcopy(self.tree_node)

        if k is not None:
            while nodes_dict[root].child_h > k:
                diff, node_id, p_id = heapq.heappop(cmp_heap)
                if child_tree_deepth(nodes_dict, node_id) <= k:
                    continue
                children = nodes_dict[node_id].children
                compressNode(nodes_dict, node_id, p_id)
                self.SE += diff
                if nodes_dict[root].child_h == k:
                    break
                for e in cmp_heap:
                    if e[1] == p_id:
                        if child_tree_deepth(nodes_dict, p_id) > k:
                            e[0] = self.CompressDelta(nodes_dict[e[1]], nodes_dict[e[2]], g_vol)
                    if e[1] in children:
                        if nodes_dict[e[1]].child_h == 0:
                            continue
                        if child_tree_deepth(nodes_dict, e[1]) > k:
                            e[2] = p_id
                            e[0] = self.CompressDelta(nodes_dict[e[1]], nodes_dict[p_id], g_vol)
                heapq.heapify(cmp_heap)
        return root, tree_node_copy

    def get_partition_from_2d(self, n_instances):
        root_id = self.root_id
        y_pred = np.zeros(n_instances)
        children = self.tree_node[root_id].children
        cluster_nodeIDs = []
        for index, childID in enumerate(children):
            cluster_nodeIDs.append(childID)
            child = self.tree_node[childID]
            partition = child.partition
            for vertex in partition:
                y_pred[vertex] = childID
        return y_pred, cluster_nodeIDs

    def get_partition_largest_clusters(self, n_cluster, n_instance, hierarchical_tree_node):
        root_id = self.root_id
        # y_pred = np.zeros(n_instance)
        # cluster_nodeIDs = []
        queue = Queue()
        queue.put(root_id)
        while queue.qsize()<n_cluster:
            node_id = queue.get()
            node = hierarchical_tree_node[node_id]
            if node.children is None:
                # cluster_nodeIDs.append(node_id)
                queue.put(node_id)
            else:
                for child_id in node.children:
                    queue.put(child_id)
        cluster_nodeIDs = []
        while queue.qsize()>0:
            cluster_nodeIDs.append(queue.get())
        y_pred = np.zeros(n_instance)
        for cluster_nodeID in cluster_nodeIDs:
            node = hierarchical_tree_node[cluster_nodeID]
            partition = node.partition
            for vertex in partition:
                y_pred[vertex] = cluster_nodeID
        return y_pred, cluster_nodeIDs


    def build_tree(self, n_cluster=None, k=None):
        self.root_id, hierarchical_tree_node = self.__build_k_tree(self.VOL, self.tree_node, k=k)
        if k == None:
            return self.root_id, hierarchical_tree_node
        y_pred, cluster_nodeIDs = self.get_partition_from_2d(self.adj_matrix.shape[0])
        return self.root_id, hierarchical_tree_node, y_pred, cluster_nodeIDs

    def subtree2etetree(self, node_id, tree_node_dict):
        t_ete = Tree(name=node_id)
        queue = Queue()
        queue.put(t_ete)
        while not queue.empty():
            t_ete_node = queue.get()
            nodeID = int(t_ete_node.name)
            tree_node = tree_node_dict[nodeID]
            if tree_node.children is not None:
                for child in tree_node.children:
                    if tree_node_dict[child].children is not None:
                        t_ete_child = t_ete_node.add_child(name=child)
                        queue.put(t_ete_child)
                    else:
                        t_ete_child = t_ete_node.add_child(name=child)
        return t_ete


