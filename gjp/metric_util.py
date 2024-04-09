import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import jraph
import networkx as nx
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import linen as nn
from networkx.drawing.nx_pydot import to_pydot

from .graphset import GraphData, batch_list, convert_to_jraph
from .model import MessagePassing


def nx_from_jraph(jraph_graph):
    nx_graph = nx.MultiDiGraph()
    for i, feature in enumerate(jraph_graph.nodes):
        nx_graph.add_node(i, feature=feature)

    for u, v, feature in zip(jraph_graph.senders, jraph_graph.receivers, jraph_graph.edges):
        nx_graph.add_edge(u_for_edge=int(u), v_for_edge=int(v), feature=feature)
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
    pydot_graph.write_svg(filename)


def _loss_helper(x):
    # return (jnp.exp(nn.relu(-x + 1)) - 1) / (jnp.exp(1) - 1) * 100
    return jnp.exp(-x * 2) * 100


def loss_function_where(params, graph, model, threshold):
    out_graph = model.apply(params, graph)
    metric_embeds = out_graph.globals[:-1]

    dist_matrix = jnp.sqrt(jnp.sum((metric_embeds[:, None] - metric_embeds[None, :]) ** 2, axis=-1))
    clean_matrix = jnp.fill_diagonal(dist_matrix, jnp.nan, inplace=False)

    idx = jnp.where(clean_matrix < threshold)
    return idx


def loss_function_combined(params, graph, model):
    out_graph = model.apply(params, graph)
    metric_embeds = out_graph.globals[:-1]

    dist_matrix = jnp.sum((metric_embeds[:, None] - metric_embeds[None, :]) ** 2, axis=-1)
    clean_matrix = jnp.fill_diagonal(dist_matrix, jnp.nan, inplace=False)

    n = metric_embeds.shape[0]
    matrix_before_sum = _loss_helper(clean_matrix)
    mean = jnp.nansum(matrix_before_sum) / (n * (n - 1))

    return mean


def loss_function_single(params, graph, model):
    out_graph = model.apply(params, graph)
    metric_embeds = out_graph.globals[:-1]

    dist_matrix = jnp.sum((metric_embeds[:, None] - metric_embeds[None, :]) ** 2, axis=-1)
    clean_matrix = jnp.fill_diagonal(dist_matrix, jnp.nan, inplace=False)
    mean = jnp.nanmin(clean_matrix)

    return _loss_helper(mean)


def train_model(train_batch, batch_test, steps, loss_grad_fn, jit_loss, jit_loss_single, params, tx, opt_state):

    @jax.jit
    def inner(i, val):
        params, opt_state, train_loss, train_max = val
        train_loss = 0
        train_max = 0
        for train_graph in train_batch:
            # Summed loss
            train_loss_val, grads = loss_grad_fn(params, train_graph)
            train_loss += train_loss_val
            updates, opt_state = tx.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            # Max loss
            train_loss_max = jit_loss_single(params, train_graph)
            train_max = jnp.max(jnp.array([train_max, train_loss_max]))

        train_loss /= len(train_batch)
        return params, opt_state, train_loss, train_max

    (params, opt_state, train_loss, train_max) = jax.lax.fori_loop(0, steps, inner, init_val=(params, opt_state, 0, 0))
    print("train loss", train_loss, train_max, jit_loss(params, batch_test), jit_loss_single(params, batch_test))

    return params, tx, opt_state


def run_parameter(
    shelf_path,
    mlp_stack,
    stepA,
    stepB,
    min_nodes=3,
    max_nodes=50,
    train_size=2500,
    test_size=500,
    extra_feature=5,
    num_batch_shuffle=5,
    seed=None,
    node_pad=20000,
    edge_pad=40000,
    learning_rate=1e-2,
    checkpoint_path=None,
    checkpoint_every=None,
    norm=None,
    from_checkpoint=False,
    epoch_offset: int = 0,
):

    print("shelf_path", shelf_path)
    print("mlp_stack", mlp_stack)
    print("stepA", stepA)
    print("stepB", stepB)
    print("min_nodes", min_nodes)
    print("max_nodes", max_nodes)
    print("train_size", train_size)
    print("test_size", test_size)
    print("extra_feature", extra_feature)
    print("num_batch_shuffle", num_batch_shuffle)
    print("seed", seed)
    print("node_pad", node_pad)
    print("edge_pad", edge_pad)
    print("learning_rate", learning_rate)
    print("checkpoint_every", checkpoint_every)
    print("checkpoint_path", checkpoint_path)
    print("from_checkpoint", from_checkpoint)
    print("epoch_offset", epoch_offset)
    print("norm", norm)
    if not norm:
        norm = [False] * len(mlp_stack)

    orbax_checkpointer = ocp.PyTreeCheckpointer()

    with GraphData(shelf_path, seed=seed) as dataset:
        train, test = dataset.get_test_train(train_size, test_size, min_nodes, max_nodes)
        node_batch_size = node_pad
        edge_batch_size = edge_pad

        train_jraph = convert_to_jraph(train)
        test_jraph = convert_to_jraph(test)

        # Add graphs with the same architecture but different initial features
        similar_train_graphs = dataset.get_similar_feature_graphs(train_jraph[0], train_size // extra_feature)
        similar_test_graphs = dataset.get_similar_feature_graphs(test_jraph[0], test_size // extra_feature)

        train_jraph += similar_train_graphs
        test_jraph += similar_test_graphs

        # Batch the test into a single graph
        batch_test = batch_list(test_jraph, node_batch_size, edge_batch_size)
        # batch_test = change_global_jraph_to_props(batch_test, node_batch_size)
        if len(batch_test) != 1:
            print("WARNING: test set doesn't fit in a single batch")
        batch_test = batch_test[0]

        # Use different combinations of batching the training data
        np_rng = np.random.default_rng(seed)
        batch_shuffles = []
        for _ in range(num_batch_shuffle):
            batched_train_data = batch_list(train_jraph, node_batch_size, edge_batch_size)
            # batched_train_data = change_global_jraph_to_props(batched_train_data, node_batch_size)

            batch_shuffles.append(batched_train_data)
            np_rng.shuffle(train_jraph)

        node_stack = mlp_stack
        edge_stack = mlp_stack
        global_stack = mlp_stack

        model = MessagePassing(edge_stack, node_stack, global_stack, num_nodes=node_batch_size, norm_global=norm)
        rng = jax.random.key(np_rng.integers(50000))
        rng, init_rng = jax.random.split(rng)
        if not from_checkpoint:
            params = model.init(init_rng, batch_test)
        else:
            params = orbax_checkpointer.restore(os.path.abspath(from_checkpoint))

        tx = optax.adam(learning_rate=learning_rate)
        opt_state = tx.init(params)

        train_loss = partial(loss_function_combined, model=model)
        test_loss = partial(loss_function_single, model=model)

        jit_loss = jax.jit(train_loss)
        jit_loss_single = jax.jit(test_loss)
        loss_grad_fn = jax.jit(jax.value_and_grad(train_loss))

        # Learning loop
        for i in range(stepA):
            if checkpoint_path and checkpoint_every and (epoch_offset + i) % checkpoint_every == 0:
                orbax_checkpointer.save(os.path.abspath(checkpoint_path + f"{epoch_offset+i}"), params)

            start = time.time()
            params, tx, opt_state = train_model(batch_shuffles[i % num_batch_shuffle], batch_test, stepB, loss_grad_fn, jit_loss, jit_loss_single, params, tx, opt_state)
            end = time.time()
            print(i + epoch_offset, end - start)

        if checkpoint_path:
            orbax_checkpointer.save(os.path.abspath(checkpoint_path + f"{epoch_offset+stepA}"), params)

        return params
