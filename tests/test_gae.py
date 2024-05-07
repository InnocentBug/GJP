from functools import partial

import jax
import jax.numpy as jnp
import jraph
import optax
import pytest

from gjp import (
    GraphData,
    MessagePassing,
    batch_list,
    convert_to_jraph,
    gae,
    metric_util,
)


def test_gae():
    with GraphData(".test_batching") as dataset:
        train, _ = dataset.get_test_train(15, 0, 5, 11)
        train_jraph = convert_to_jraph(train)
        for i in range(15):
            similar_data = dataset.get_similar_feature_graphs(train_jraph[i], 2)
            train_jraph += similar_data

        num_nodes_train, num_edges_train = metric_util._count_nodes_edges(train_jraph)

        node_batch_size = num_nodes_train + 1
        edge_batch_size = num_edges_train + 1
        batch_train = batch_list(train_jraph, node_batch_size, edge_batch_size)[0]

        max_num_nodes = 20
        max_num_edges = 40
        encoder_stack = [[2], [4, 8], [16], [2]]

        model = gae.GAE(
            encoder_stack=encoder_stack,
            max_num_nodes=max_num_nodes,
            max_num_edges=max_num_edges,
            init_stack=[8, 4, 16, 32],
            init_features=4,
            prob_stack=[[4], [8], [16]],
            feature_stack=[[4], [8], [16]],
            node_features=batch_train.nodes.shape[1],
            edge_features=batch_train.edges.shape[1],
            max_num_graph=len(train_jraph),
        )
        rng = jax.random.key(234)
        rng, rng_split = jax.random.split(rng)

        params = model.init({"params": rng, "reparametrize": rng_split}, batch_train)
        rng, rng_split = jax.random.split(rng)

        apply_model = jax.jit(lambda x, y, z: model.apply(x, y, rngs={"reparametrize": z}))
        out_graphs = apply_model(params, batch_train, rng_split)
        rng, rng_split = jax.random.split(rng)

        assert out_graphs.nodes.shape == (batch_train.n_node.shape[0] * max_num_nodes, batch_train.nodes.shape[1])
        assert out_graphs.edges.shape == (batch_train.n_node.shape[0] * max_num_edges, batch_train.edges.shape[1])

        assert out_graphs.senders.shape[0] == batch_train.n_node.shape[0] * max_num_edges
        assert out_graphs.receivers.shape[0] == batch_train.n_node.shape[0] * max_num_edges

        assert out_graphs.n_node.shape[0] == 2 * batch_train.n_node.shape[0]
        assert out_graphs.n_edge.shape[0] == 2 * batch_train.n_edge.shape[0]

        assert jnp.sum(out_graphs.n_node) == batch_train.n_node.shape[0] * max_num_nodes
        assert jnp.sum(out_graphs.n_edge) == batch_train.n_edge.shape[0] * max_num_edges

        assert out_graphs.globals.shape == (2 * batch_train.n_node.shape[0], encoder_stack[-1][-1] + 2)

        out_unbatch = jraph.unbatch(out_graphs)

        for i, in_graph in enumerate(train_jraph):
            out_graph = out_unbatch[2 * i]
            assert out_graph.n_node == in_graph.n_node
            assert out_graph.n_edge[0] <= in_graph.n_edge[0]


@pytest.mark.parametrize("norm", [True, False])
def test_loss(norm):
    with GraphData(".test_batching") as dataset:
        train, _ = dataset.get_test_train(15, 0, 5, 11)
        train_jraph = convert_to_jraph(train)
        for i in range(15):
            similar_data = dataset.get_similar_feature_graphs(train_jraph[i], 2)
            train_jraph += similar_data

        num_nodes_train, num_edges_train = metric_util._count_nodes_edges(train_jraph)

        node_batch_size = num_nodes_train + 1
        edge_batch_size = num_edges_train + 1
        batch_train = batch_list(train_jraph, node_batch_size, edge_batch_size)[0]

        max_num_nodes = 20
        max_num_edges = 40
        encoder_stack = [[2], [4, 8], [16], [2]]

        model = gae.GAE(
            encoder_stack=encoder_stack,
            max_num_nodes=max_num_nodes,
            max_num_edges=max_num_edges,
            init_stack=[8, 4, 16, 32],
            init_features=4,
            prob_stack=[[4], [8], [16]],
            feature_stack=[[4], [8], [16]],
            node_features=batch_train.nodes.shape[1],
            edge_features=batch_train.edges.shape[1],
            max_num_graph=len(train_jraph),
        )
        rng = jax.random.key(234)
        rng, rng_split = jax.random.split(rng)

        params = model.init({"params": rng, "reparametrize": rng_split}, batch_train)
        rng, rng_split = jax.random.split(rng)

        final_node_size = 21
        final_edge_size = 34
        final_global_size = 42

        node_stack = [[4], [2], [final_node_size]]
        edge_stack = [[4], [2], [final_edge_size]]
        global_stack = [[4], [2], [final_global_size]]
        metric_model = MessagePassing(node_stack, edge_stack, global_stack, num_nodes=max_num_nodes * len(train_jraph))
        metric_params = metric_model.init(rng_split, batch_train)

        partial_loss = partial(gae.loss_function, model=model, metric_params=metric_params, metric_model=metric_model, norm=norm)

        func = jax.jit(jax.value_and_grad(partial_loss))

        tx = optax.adam(learning_rate=1e-3)
        opt_state = tx.init(params)
        loss, grads = func(params, batch_train, rng_split)
        updates, opt_state = tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
