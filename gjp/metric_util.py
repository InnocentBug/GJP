import jax.numpy as jnp
import jraph
import networkx as nx
from networkx.drawing.nx_pydot import to_pydot


def nx_from_jraph(jraph_graph):
    nx_graph = nx.MultiDiGraph()
    counter = 0
    for i, feature in enumerate(jraph_graph.nodes):
        nx_graph.add_node(i, feature=feature)
        counter += 1

    edge_counter = 0
    for u, v, feature in zip(jraph_graph.senders, jraph_graph.receivers, jraph_graph.edges):
        nx_graph.add_edge(u_for_edge=int(u), v_for_edge=int(v), feature=feature)
        edge_counter += 1

    return nx_graph


def svg_graph_list(graphs, filename="graphs.svg"):
    nx_graphs = []
    for g in graphs:
        if isinstance(g, jraph.GraphsTuple):
            nx_graphs.append(nx_from_jraph(g))
        else:
            nx_graphs.append(g)
    combined_graphs = nx.disjoint_union_all(nx_graphs)
    pydot_graph = to_pydot(combined_graphs)
    pydot_graph.write_pdf(filename)


def _loss_helper(x):
    return jnp.exp(-x * 2) * 100


def loss_function_where(params, graph, model, threshold, rngs=None):
    out_graph = model.apply(params, graph, rngs=rngs)
    metric_embeds = out_graph.globals[:-1]

    dist_matrix = jnp.sqrt(jnp.sum((metric_embeds[:, None] - metric_embeds[None, :]) ** 2, axis=-1))
    clean_matrix = jnp.fill_diagonal(dist_matrix, jnp.nan, inplace=False)

    idx = jnp.where(clean_matrix < threshold)
    return idx


def loss_function_where_num(params, graph, model, num, rngs=None):
    out_graph = model.apply(params, graph, rngs=rngs)
    metric_embeds = out_graph.globals[:-1]

    dist_matrix = jnp.sqrt(jnp.sum((metric_embeds[:, None] - metric_embeds[None, :]) ** 2, axis=-1))
    clean_matrix = jnp.fill_diagonal(dist_matrix, jnp.inf, inplace=False)

    argsort_flat = jnp.argsort(clean_matrix, axis=None)
    indeces = []
    for i in range(num):
        idx = argsort_flat[2 * i]
        a = int(idx // clean_matrix.shape[0])
        b = int(idx % clean_matrix.shape[1])
        indeces += [(a, b)]
    return indeces


def loss_function_combined(params, graph, model, rngs=None, norm=False, norm_step=4.605170186):
    out_graph = model.apply(params, graph, rngs=rngs)
    metric_embeds = out_graph.globals[:-1]

    dist_matrix = jnp.sum((metric_embeds[:, None] - metric_embeds[None, :]) ** 2, axis=-1)
    clean_matrix = jnp.fill_diagonal(dist_matrix, jnp.nan, inplace=False)

    n = metric_embeds.shape[0]
    matrix_before_sum = _loss_helper(clean_matrix)
    mean = jnp.nansum(matrix_before_sum) / (n * (n - 1))

    if norm:
        mean += 10 * jnp.exp(-norm_step) * jnp.mean(metric_embeds**2)

    return mean


def loss_function_single(params, graph, model, rngs=None):
    out_graph = model.apply(params, graph)
    metric_embeds = out_graph.globals[:-1]

    dist_matrix = jnp.sum((metric_embeds[:, None] - metric_embeds[None, :]) ** 2, axis=-1)
    clean_matrix = jnp.fill_diagonal(dist_matrix, jnp.nan, inplace=False)
    mean = jnp.nanmin(clean_matrix)

    return _loss_helper(mean)


def _count_nodes_edges(graph_list):
    num_nodes = 0
    num_edges = 0
    for graph in graph_list:
        num_nodes += jnp.sum(graph.n_node)
        num_edges += jnp.sum(graph.n_edge)
    return num_nodes, num_edges


def count_nodes_edges(graph_list):
    return _count_nodes_edges(graph_list)
