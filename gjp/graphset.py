import copy
import shelve
import time
from typing import Optional

import jax
import jax.numpy as jnp
import jraph
import networkx as nx
import numpy as np


class GraphData:
    graph_types = ("similar", "random")

    def __init__(self, shelve_path_root: str, seed: Optional[int] = None, overwrite=False, min_edges=None, max_edges=None):
        self._shelve_path_root = shelve_path_root
        self._shelve_handle = None
        if overwrite:
            for gt in self.graph_types:
                with shelve.open(self._shelve_path_root + "_" + gt, "n"):
                    pass
        self._min_edges = min_edges
        self._max_edges = max_edges

        if seed is None:
            seed = np.int32(time.time())
        self._np_rng = np.random.default_rng(seed=seed)
        self._jax_rng = jax.random.key(seed)

    def __enter__(self):
        self._shelve_handle = {}
        for gt in self.graph_types:
            self._shelve_handle[gt] = shelve.open(self._shelve_path_root + "_" + gt, "c")

        return self

    def __exit__(self, type, value, traceback):
        for graph_type in self._shelve_handle:
            self._shelve_handle[graph_type].close()

    def _ensure_working_shelf(self, num_nodes):
        for graph_type in self.graph_types:
            try:
                _ = self._shelve_handle[graph_type][str(num_nodes)]
            except KeyError:
                self._shelve_handle[graph_type][str(num_nodes)] = []

    def check_graph_isomorph(self, graph):
        self._ensure_working_shelf(len(graph))

        found_isomorph = False
        num_nodes = str(len(graph))
        for graph_type in self._shelve_handle.keys():
            if not found_isomorph:
                for existing_graph in self._shelve_handle[graph_type][str(num_nodes)]:
                    if nx.is_isomorphic(graph, existing_graph):
                        found_isomorph = True
                        break
        return found_isomorph

    def generate_random_directed_graph(self, num_nodes):
        self._ensure_working_shelf(num_nodes)

        # Create an empty directed graph
        graph = nx.MultiDiGraph()

        # Add nodes to the graph
        for i in range(num_nodes):
            graph.add_node(i)
            graph.add_node(i)

        # 2 because of directed graph
        max_edges = self._max_edges
        if max_edges is None:
            max_edges = 2 * num_nodes
        min_edges = self._min_edges
        if min_edges is None:
            min_edges = num_nodes

        num_edges = self._np_rng.integers(low=min_edges, high=max_edges)
        while len(graph.edges()) < num_edges:
            u, v = self._np_rng.choice(range(num_nodes), 2)
            graph.add_edge(u, v)

        found_isomorph = self.check_graph_isomorph(graph)
        if not found_isomorph:
            self._shelve_handle["random"][str(num_nodes)] += [graph]

        return not found_isomorph

    def ensure_num_random_graphs(self, num_nodes, n_graphs):
        self._ensure_working_shelf(num_nodes)
        while self.num_graphs("random", num_nodes) < n_graphs:
            self.generate_random_directed_graph(num_nodes)

    def generate_similar_directed_graph(self, num_nodes):
        self._ensure_working_shelf(num_nodes)

        old_graph = None
        if self.num_graphs("similar", num_nodes) > 0:
            idx = self._np_rng.choice(len(self._shelve_handle["similar"][str(num_nodes)]))
            old_graph = self.get_graph("similar", num_nodes, idx)
        else:
            if self.num_graphs("random", num_nodes) == 0:
                self.ensure_num_random_graphs(num_nodes, 1)
            idx = self._np_rng.choice(self.num_graphs("random", num_nodes))
            old_graph = self.get_graph("random", num_nodes, idx)

        new_graph = copy.copy(old_graph)

        change_similar_mode = self._np_rng.integers(3)

        if change_similar_mode == 0:  # Add an edge
            u, v = self._np_rng.choice(range(num_nodes), 2)
            new_graph.add_edge(u, v)

        if change_similar_mode == 1:  # Remove an edge
            if len(new_graph.edges()) <= num_nodes:
                return False
            edge = self._np_rng.choice(list(new_graph.edges()))
            new_graph.remove_edge(*edge)

        if change_similar_mode == 2:
            if len(new_graph.edges()) == 0:
                return False
            edge = self._np_rng.choice(list(new_graph.edges()))
            new_graph.remove_edge(*edge)
            if self._np_rng.integers(2) == 0:
                new_edge = (edge[0], self._np_rng.choice(range(num_nodes)))
            else:
                new_edge = (self._np_rng.choice(range(num_nodes)), edge[1])
            new_graph.add_edge(*new_edge)

        found_isomorph = self.check_graph_isomorph(new_graph)
        if not found_isomorph:
            self._shelve_handle["similar"][str(num_nodes)] += [new_graph]

        return not found_isomorph

    def ensure_num_similar_graphs(self, num_nodes, n_graphs):
        self._ensure_working_shelf(num_nodes)
        while self.num_graphs("similar", num_nodes) < n_graphs:
            self.generate_similar_directed_graph(num_nodes)

    def get_graph(self, mode: str, num_nodes, idx):
        return self._shelve_handle[mode][str(num_nodes)][idx]

    def num_graphs(self, mode: str, num_nodes: int):
        self._ensure_working_shelf(num_nodes)
        return len(self._shelve_handle[mode][str(num_nodes)])

    def get_test_train(self, train_size: int, test_size: int, min_nodes: int, max_nodes: int):
        all_nodes = []
        for num_nodes in range(min_nodes, max_nodes):
            self.ensure_num_random_graphs(num_nodes, test_size + train_size)
            for idx in range(self.num_graphs("random", num_nodes)):
                all_nodes.append(self.get_graph("random", num_nodes, idx))

            self.ensure_num_similar_graphs(num_nodes, test_size + train_size)
            for idx in range(self.num_graphs("similar", num_nodes)):
                all_nodes.append(self.get_graph("similar", num_nodes, idx))

        self._np_rng.shuffle(all_nodes)

        return all_nodes[:train_size], all_nodes[train_size : train_size + test_size]

    def get_similar_feature_graphs(self, jraph_graph, num_graphs):
        similar_graphs = []
        while len(similar_graphs) < num_graphs:
            if self._np_rng.integers(2) == 0:
                node_idx = self._np_rng.integers(jraph_graph.nodes.shape[0])
                new_nodes = jraph_graph.nodes.at[node_idx].set(self._np_rng.normal())
                new_graph = jraph_graph._replace(nodes=new_nodes)
            else:
                edge_idx = self._np_rng.integers(jraph_graph.edges.shape[0])
                new_edges = jraph_graph.edges.at[edge_idx].set(self._np_rng.normal())
                new_graph = jraph_graph._replace(edges=new_edges)
            similar_graphs += [new_graph]
        self._np_rng.shuffle(similar_graphs)
        return similar_graphs


def convert_to_jraph(graphs, num_nodes_pad: int = None, num_edges_pad: int = None):
    try:
        iter(graphs)
    except TypeError:
        graphs = [graphs]

    jraph_graphs = []
    for nx_graph in graphs:
        senders = []
        receivers = []
        for edge in nx_graph.edges(data=True):
            senders += [edge[0]]
            receivers += [edge[1]]
        if num_edges_pad and len(senders) >= num_edges_pad:
            raise RuntimeError("Number of edges larger then request pad size for edges.")
        edges = jnp.zeros((len(senders), 1))
        if num_nodes_pad and len(nx_graph) >= num_nodes_pad:
            raise RuntimeError("Number of nodes larger then request pad size for nodes.")
        node_features = jnp.zeros((len(nx_graph), 1))
        global_context = jnp.array([[1]])
        graph = jraph.GraphsTuple(
            nodes=jnp.asarray(node_features),
            edges=jnp.asarray(edges).reshape(len(edges), 1),
            senders=jnp.asarray(senders, dtype=int),
            receivers=jnp.asarray(receivers, dtype=int),
            n_node=jnp.asarray([len(node_features)]),
            n_edge=jnp.asarray([len(senders)]),
            globals=global_context,
        )
        if num_nodes_pad is not None and num_edges_pad is not None:
            pad_graph = get_pad_graph(graph, num_nodes_pad, num_edges_pad)
            jraph_graphs.append(jraph.batch([graph, pad_graph]))
        else:
            jraph_graphs.append(graph)
    return jraph_graphs


def get_pad_graph(graph, num_nodes_pad, num_edges_pad):
    return get_pad_graph_internal(graph.nodes.shape, graph.edges.shape, graph.globals.shape, num_nodes_pad, num_edges_pad)


def get_pad_graph_internal(nodes_shape, edges_shape, globals_shape, num_nodes_pad, num_edges_pad):
    pad_graph = jraph.GraphsTuple(
        nodes=jnp.zeros((num_nodes_pad - nodes_shape[0],) + nodes_shape[1:]),
        edges=jnp.zeros((num_edges_pad - edges_shape[0],) + edges_shape[1:]),
        senders=jnp.zeros(num_edges_pad - edges_shape[0], dtype=int),
        receivers=jnp.zeros(num_edges_pad - edges_shape[0], dtype=int),
        n_node=jnp.asarray([num_nodes_pad - nodes_shape[0]]),
        n_edge=jnp.asarray([num_edges_pad - edges_shape[0]]),
        globals=jnp.zeros((1,) + globals_shape[1:]),
    )

    return pad_graph


def batch_list(graph_list, batch_nodes, batch_edges):
    batch_list = []
    tmp_list = []
    num_nodes = 0
    num_edges = 0
    for graph in graph_list:
        if graph.nodes.shape[0] + num_nodes >= batch_nodes or graph.edges.shape[0] + num_edges >= batch_edges:

            batched_graph = jraph.batch(tmp_list)
            pad_graph = get_pad_graph(batched_graph, batch_nodes, batch_edges)
            batch_list.append(jraph.batch([batched_graph, pad_graph]))

            tmp_list = []
            num_edges = 0
            num_nodes = 0

        tmp_list += [graph]
        num_nodes += graph.nodes.shape[0]
        num_edges += graph.edges.shape[0]

    if len(tmp_list) > 0:
        batched_graph = jraph.batch(tmp_list)
        pad_graph = get_pad_graph(batched_graph, batch_nodes, batch_edges)
        batch_list.append(jraph.batch([batched_graph, pad_graph]))

    return batch_list
