import copy
import itertools
import os
import shelve
import time
from functools import partial
from typing import Optional, Union

import jax
import jax.numpy as jnp
import jraph
import networkx as nx
import numpy as np
import optax
import pydot
from networkx.drawing.nx_pydot import graphviz_layout, to_pydot

from .graphset import batch_list, get_pad_graph


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
    mean = jnp.nansum(clean_matrix) / (n * (n - 1))

    return 1 / (1 + mean)


def loss_function_single(params, graph, model):
    out_graph = model.apply(params, graph)
    metric_embeds = out_graph.globals[:-1]

    dist_matrix = jnp.sqrt(jnp.sum((metric_embeds[:, None] - metric_embeds[None, :]) ** 2, axis=-1))
    clean_matrix = jnp.fill_diagonal(dist_matrix, jnp.nan, inplace=False)
    mean = jnp.nanmin(clean_matrix)
    return 1 / (1 + mean)


def train_model(train_batch, batch_test, steps, model, params, tx, opt_state):

    train_loss = partial(loss_function_combined, model=model)
    test_loss = partial(loss_function_single, model=model)

    jit_loss = jax.jit(train_loss)
    jit_loss_single = jax.jit(test_loss)
    loss_grad_fn = jax.jit(jax.value_and_grad(train_loss))

    @jax.jit
    def inner(i, val):
        params, opt_state, train_loss = val
        train_loss = 0
        for train_graph in train_batch:
            train_loss_val, grads = loss_grad_fn(params, train_graph)
            train_loss += train_loss_val
            updates, opt_state = tx.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
        train_loss /= len(train_batch)
        return params, opt_state, train_loss

    (
        params,
        opt_state,
        train_loss,
    ) = jax.lax.fori_loop(0, steps, inner, init_val=(params, opt_state, 0))
    print("train loss", train_loss, jit_loss(params, batch_test), jit_loss_single(params, batch_test))

    return params, tx, opt_state
