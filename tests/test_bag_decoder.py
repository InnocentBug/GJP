import hashlib
import os

import jax
import jax.numpy as jnp
import jraph
import pytest
from flax import linen as nn

from gjp import bag_decoder, metric_util, mpg

MLP_KWARGS = {"dropout_rate": 0.1, "deterministic": False, "activation": nn.sigmoid}


@pytest.mark.parametrize("max_num_nodes, stack", [(5, [15, 76, 65, 1]), (6, [15, 7]), (10, [2])])
def test_initial_node_decoder(max_num_nodes, stack, jax_rng):
    print("")
    graph_num = 3
    rng = jax_rng
    for _ in range(max_num_nodes):
        rng, _ = jax.random.split(rng)

    rng, init_rng, dropout_rng, node_rng = jax.random.split(rng, 4)
    n_node_edge = jax.random.randint(node_rng, (graph_num, 2), 1, max_num_nodes - 1)
    model = bag_decoder.InitialNodeBag(max_nodes=max_num_nodes, mlp_size=stack, mlp_kwargs=MLP_KWARGS)

    rng, rng_normal = jax.random.split(rng)
    test_input = jax.random.normal(rng_normal, (graph_num, 7))
    test_input = jnp.hstack((test_input, n_node_edge))

    params = model.init({"params": init_rng, "dropout": dropout_rng}, test_input)
    rng, dropout_rng = jax.random.split(rng)

    apply_model = jax.jit(model.apply)
    apply_model(params, test_input, rngs={"dropout": dropout_rng})
    rng, dropout_rng = jax.random.split(rng)
    initial_node_features = apply_model(params, test_input, rngs={"dropout": dropout_rng})
    rng, dropout_rng = jax.random.split(rng)

    assert initial_node_features.shape == (test_input.shape[0], max_num_nodes, stack[-1])
    for i in range(graph_num):
        n_node = n_node_edge[i, 0]
        first_half = initial_node_features[i, :n_node, :]
        assert jnp.mean(jnp.abs(first_half)) > 0
        second_half = initial_node_features[i, n_node:, :]
        assert jnp.sum(jnp.abs(second_half)) == 0


@pytest.mark.parametrize("max_num_nodes, multi_edge_repeat, stack", [(5, 1, [15, 76, 65, 1]), (6, 2, [15, 7]), (10, 3, [2])])
def test_full_edge_decoder(max_num_nodes, multi_edge_repeat, stack, jax_rng):
    graph_num = 3
    rng = jax_rng
    for _ in range(max_num_nodes):
        rng, _ = jax.random.split(rng)

    rng, init_rng, dropout_rng, node_rng = jax.random.split(rng, 4)
    n_node_edge = jax.random.randint(node_rng, (graph_num, 2), 2, max_num_nodes - 1)
    model = bag_decoder.InitialBagEdges(max_nodes=max_num_nodes, multi_edge_repeat=multi_edge_repeat, mlp_size=stack, mlp_kwargs=MLP_KWARGS)

    rng, rng_normal = jax.random.split(rng)
    test_input = jax.random.normal(rng_normal, (graph_num, 7))
    test_input = jnp.hstack((test_input, n_node_edge))

    params = model.init({"params": init_rng, "dropout": dropout_rng}, test_input)
    rng, dropout_rng = jax.random.split(rng)

    apply_model = jax.jit(model.apply)
    apply_model(params, test_input, rngs={"dropout": dropout_rng})
    rng, dropout_rng = jax.random.split(rng)
    senders, receivers, features = apply_model(params, test_input, rngs={"dropout": dropout_rng})

    assert senders.shape == (graph_num * max_num_nodes**2 * multi_edge_repeat,)
    assert receivers.shape == (graph_num * max_num_nodes**2 * multi_edge_repeat,)
    assert features.shape == (graph_num * max_num_nodes**2 * multi_edge_repeat, stack[-1])

    senders = senders.reshape(
        (
            graph_num,
            max_num_nodes**2 * multi_edge_repeat,
        )
    )
    receivers = receivers.reshape(
        (
            graph_num,
            max_num_nodes**2 * multi_edge_repeat,
        )
    )

    for i in range(graph_num):
        g_senders = senders[i]
        assert jnp.min(g_senders) == i * max_num_nodes
        assert jnp.max(g_senders) <= (i + 1) * max_num_nodes - 1

        g_senders = g_senders - i * max_num_nodes

        g_receivers = receivers[i]
        assert jnp.min(g_receivers) == i * max_num_nodes
        assert jnp.max(g_receivers) <= (i + 1) * max_num_nodes - 1

        g_receivers = g_receivers - i * max_num_nodes

        print(i, g_senders)
        print(i, g_receivers)


@pytest.mark.parametrize("max_num_nodes, multi_edge_repeat, stack, mpg_stack", [(5, 1, [15, 76, 65, 1], [[2], [2, 4]]), (6, 2, [15, 7], [[11], [2, 3], [3]]), (10, 3, [2], [[2], [4], [5], [3], [2]])])
def test_init_bag_graph(max_num_nodes, multi_edge_repeat, stack, mpg_stack, jax_rng, ensure_tempfile):
    print("")
    graph_num = 4
    rng = jax_rng
    for _ in range(graph_num + max_num_nodes):
        rng, _ = jax.random.split(rng)

    rng, init_rng, dropout_rng, node_rng = jax.random.split(rng, 4)
    n_node = jax.random.randint(node_rng, (graph_num,), 2, max_num_nodes - 1)
    n_edge = []
    for n in n_node:
        n_edge.append(jax.random.randint(node_rng, (1,), 2, (n) ** 2 * multi_edge_repeat)[0])
        rng, node_rng = jax.random.split(rng)
    n_edge = jnp.asarray(n_edge, dtype=int)
    n_node_edge = jnp.vstack((n_node, n_edge)).transpose()

    model = bag_decoder.InitialGraphBagDecoder(max_nodes=max_num_nodes, init_node_stack=stack, init_edge_stack=stack, message_passing_stack=mpg_stack, multi_edge_repeat=multi_edge_repeat, mlp_kwargs=MLP_KWARGS)

    rng, rng_normal = jax.random.split(rng)
    test_input = jax.random.normal(rng_normal, (graph_num, 7))
    test_input = jnp.hstack((test_input, n_node_edge))

    params = model.init({"params": init_rng, "dropout": dropout_rng}, test_input)
    rng, dropout_rng = jax.random.split(rng)

    apply_model = jax.jit(model.apply)
    apply_model(params, test_input, rngs={"dropout": dropout_rng})
    rng, dropout_rng = jax.random.split(rng)
    out_nodes = apply_model(params, test_input, rngs={"dropout": dropout_rng})

    unbatch_nodes = jraph.unbatch(out_nodes)

    m = hashlib.shake_256()
    m.update(test_input.tobytes())

    original_path, _ = ensure_tempfile
    metric_util.svg_graph_list(unbatch_nodes, filename=os.path.join(original_path, f"la_graph{m.hexdigest(3)}.pdf"))

    for i, graph in enumerate(unbatch_nodes):
        if i % 2 == 0:
            vals, count = jnp.unique(graph.senders, return_counts=True)
            assert jnp.min(vals) == 0
            assert jnp.max(vals) == graph.n_node[0] - 1
            assert len(vals) == graph.n_node[0]
            assert jnp.allclose(jnp.ones(count.shape) * graph.n_node[0] * multi_edge_repeat, count)

            vals, count = jnp.unique(graph.receivers, return_counts=True)
            assert jnp.min(vals) == 0
            assert jnp.max(vals) == graph.n_node[0] - 1
            assert len(vals) == graph.n_node[0]
            assert jnp.allclose(jnp.ones(count.shape) * graph.n_node[0] * multi_edge_repeat, count)
        else:
            assert jnp.allclose(jnp.zeros(graph.senders.shape), graph.senders)
            assert jnp.allclose(jnp.zeros(graph.receivers.shape), graph.receivers)

    rng, dropout_rng = jax.random.split(rng)

    def dummy_loss(params, graph):
        out_graph = model.apply(params, graph, rngs={"dropout": dropout_rng})
        return jnp.sum(out_graph.nodes) + jnp.sum(out_graph.edges)

    # Ensure we have gradients on all our params
    grads = jax.grad(dummy_loss)(params, test_input)
    for leaf in jax.tree_util.tree_leaves(grads):
        assert jnp.sum(jnp.abs(leaf)) > 0


@pytest.mark.parametrize("max_num_nodes, multi_edge_repeat, stack, mpg_stack", [(5, 1, [15, 76, 65, 1], [[2], [2, 4]]), (6, 2, [15, 7], [[11], [2, 3], [3]]), (10, 3, [2], [[2], [4], [5], [3], [2]])])
def test_bag_graph_decode(max_num_nodes, multi_edge_repeat, stack, mpg_stack, jax_rng, ensure_tempfile):
    print("")
    graph_num = 7
    rng = jax_rng
    for _ in range(graph_num + max_num_nodes):
        rng, _ = jax.random.split(rng)

    rng, init_rng, dropout_rng, node_rng = jax.random.split(rng, 4)
    n_node = jax.random.randint(node_rng, (graph_num,), 2, max_num_nodes - 1)
    n_edge = []
    for n in n_node:
        n_edge.append(jax.random.randint(node_rng, (1,), 2, (n) ** 2 * multi_edge_repeat)[0])
        rng, node_rng = jax.random.split(rng)
    n_edge = jnp.asarray(n_edge, dtype=int)
    n_node_edge = jnp.vstack((n_node, n_edge)).transpose()

    model = bag_decoder.GraphBagDecoder(
        max_nodes=max_num_nodes, init_node_stack=stack, init_edge_stack=stack, init_mpg_stack=mpg_stack, final_mpg_node_stack=mpg_stack, final_mpg_edge_stack=mpg_stack, multi_edge_repeat=multi_edge_repeat, mlp_kwargs=MLP_KWARGS
    )

    rng, rng_normal = jax.random.split(rng)
    test_input = jax.random.normal(rng_normal, (graph_num, 7))
    test_input = jnp.hstack((test_input, n_node_edge))

    params = model.init({"params": init_rng, "dropout": dropout_rng}, test_input)
    rng, dropout_rng = jax.random.split(rng)

    apply_model = jax.jit(model.apply)
    apply_model(params, test_input, rngs={"dropout": dropout_rng})
    rng, dropout_rng = jax.random.split(rng)
    out_graphs = apply_model(params, test_input, rngs={"dropout": dropout_rng})

    unbatch_nodes = jraph.unbatch(out_graphs)

    m = hashlib.shake_256()
    m.update(test_input.tobytes())

    original_path, _ = ensure_tempfile
    metric_util.svg_graph_list(unbatch_nodes, filename=os.path.join(original_path, f"final_decode_graph{m.hexdigest(3)}.pdf"))

    for i, graph in enumerate(unbatch_nodes):
        if i % 2 == 0:
            vals, count = jnp.unique(graph.senders, return_counts=True)
            assert jnp.min(vals) >= 0
            assert jnp.max(vals) <= graph.n_node[0] - 1

            vals, count = jnp.unique(graph.receivers, return_counts=True)
            assert jnp.min(vals) >= 0
            assert jnp.max(vals) <= graph.n_node[0] - 1
        else:
            assert jnp.allclose(jnp.zeros(graph.senders.shape), graph.senders)
            assert jnp.allclose(jnp.zeros(graph.receivers.shape), graph.receivers)

    rng, dropout_rng = jax.random.split(rng)

    def dummy_loss(params, graph):
        out_graph = model.apply(params, graph, rngs={"dropout": dropout_rng})
        return jnp.sum(out_graph.nodes) + jnp.sum(out_graph.edges)

    # Ensure we have gradients on all our params
    grads = jax.grad(dummy_loss)(params, test_input)
    for leaf in jax.tree_util.tree_leaves(grads):
        assert jnp.sum(jnp.abs(leaf)) > 0

    metric_model = mpg.MessagePassingGraph(
        node_stack=mpg_stack,
        edge_stack=mpg_stack,
        attention_stack=mpg_stack,
        global_stack=mpg_stack,
    )
    rng, la_rng = jax.random.split(rng)
    metric_params = metric_model.init(la_rng, out_graphs)

    def global_loss(params, in_data):
        out_graph = model.apply(params, in_data, rngs={"dropout": dropout_rng})
        out_graph = out_graph._replace(globals=out_graph.globals * 0)

        metric_out = metric_model.apply(metric_params, out_graph)
        return jnp.mean(metric_out.globals)

    grads = jax.grad(global_loss)(params, test_input)
    for leaf in jax.tree_util.tree_leaves(grads):
        assert jnp.sum(jnp.abs(leaf)) > 0
