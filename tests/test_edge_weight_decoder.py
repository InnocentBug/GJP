import hashlib

import jax
import jax.numpy as jnp
import jraph
import pytest

from gjp import bag_gae, edge_weight_decoder, mpg_edge_weight


@pytest.mark.parametrize("max_num_nodes, multi_edge_repeat", [(5, 1), (6, 2), (10, 3)])
def test_full_edge_decoder(max_num_nodes, multi_edge_repeat, jax_rng):
    graph_num = 3

    n_node_edge = jax.random.randint(jax_rng, (graph_num, 2), 2, max_num_nodes - 1)

    rng, rng_normal = jax.random.split(jax_rng)
    test_input = jax.random.normal(rng_normal, (graph_num, 7))
    test_input = jnp.hstack((test_input, n_node_edge))

    model = edge_weight_decoder.FullyConnectedGraph(max_num_nodes, multi_edge_repeat)

    apply_model = jax.jit(model.__call__)
    apply_model(test_input)
    rng, dropout_rng = jax.random.split(rng)
    out_graph = apply_model(test_input)

    assert out_graph.senders.shape == (graph_num * max_num_nodes**2 * multi_edge_repeat,)
    assert out_graph.receivers.shape == (graph_num * max_num_nodes**2 * multi_edge_repeat,)

    unbatch_graphs = jraph.unbatch(out_graph)

    for i in range(graph_num):
        n_node = test_input[i, -2]
        graph = unbatch_graphs[2 * i]
        fill_graph = unbatch_graphs[2 * i + 1]

        assert jnp.min(graph.senders) == 0
        assert jnp.max(graph.senders) == n_node - 1

        assert jnp.min(graph.receivers) == 0
        assert jnp.max(graph.receivers) <= n_node - 1

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

        assert graph.n_node[0] == n_node
        assert graph.n_node[0] + fill_graph.n_node[0] == max_num_nodes

        assert graph.n_edge[0] == n_node**2 * multi_edge_repeat
        assert graph.n_edge[0] + fill_graph.n_edge[0] == max_num_nodes**2 * multi_edge_repeat

        assert jnp.sum(jnp.abs(fill_graph.senders)) < 1e-6
        assert jnp.sum(jnp.abs(fill_graph.receivers)) < 1e-6


def test_make_graph_fully_connected(batch_graphs):
    multi_edge_repeat = bag_gae.find_multi_edge_repeat(batch_graphs)
    max_nodes = jnp.max(batch_graphs.n_node)

    unbatch = jraph.unbatch(batch_graphs)
    for i in range(len(unbatch)):
        new_nodes = jnp.hstack([unbatch[i].nodes + i + 1, unbatch[i].nodes + i + 2])
        new_edges = jnp.hstack([unbatch[i].edges + i + 1, unbatch[i].edges + i + 2])

        unbatch[i] = unbatch[i]._replace(nodes=new_nodes, edges=new_edges)

    batch_graphs = jraph.batch(unbatch)

    fully_graph, edge_weights = edge_weight_decoder.make_graph_fully_connected(batch_graphs, multi_edge_repeat)

    assert jnp.sum(edge_weights) == jnp.sum(batch_graphs.n_edge)

    edge_weights = edge_weights.reshape((batch_graphs.n_node.shape[0], max_nodes**2 * multi_edge_repeat))

    unbatch_fully = jraph.unbatch(fully_graph)
    for i in range(len(unbatch)):
        assert jnp.allclose(unbatch[i].nodes, unbatch_fully[2 * i].nodes)
        assert jnp.sum(jnp.abs(unbatch_fully[2 * i + 1].nodes)) < 1e-12
        assert jnp.sum(jnp.abs(unbatch_fully[2 * i + 1].edges)) < 1e-12

        local_weights = edge_weights[i]
        assert jnp.max(local_weights) <= 1.0
        assert jnp.sum(local_weights) == unbatch[i].n_edge[0]

        assert jnp.sum(local_weights[: unbatch_fully[i * 2].n_edge[0]]) == unbatch[i].n_edge[0]
        assert jnp.sum(local_weights[unbatch_fully[i * 2].n_edge[0] :]) == 0

        non_zero_idx = jnp.nonzero(local_weights[: unbatch_fully[i * 2].n_edge[0]])[0]
        assert jnp.min(unbatch_fully[i * 2].edges[non_zero_idx]) > 0
        zero_idx = jnp.nonzero(~(local_weights[: unbatch_fully[i * 2].n_edge[0]]).astype(bool))[0]
        assert jnp.allclose(unbatch_fully[i * 2].edges[zero_idx], jnp.zeros((zero_idx.shape[0], unbatch_fully[i * 2].edges.shape[1])))


def test_double_conversion(batch_graphs):
    multi_edge_repeat = bag_gae.find_multi_edge_repeat(batch_graphs)
    jnp.max(batch_graphs.n_node)

    unbatch = jraph.unbatch(batch_graphs)
    for i in range(len(unbatch)):
        new_nodes = jnp.hstack([unbatch[i].nodes + i + 1, unbatch[i].nodes + i + 2])
        new_edges = jnp.hstack([unbatch[i].edges + i + 1, unbatch[i].edges + i + 2])

        unbatch[i] = unbatch[i]._replace(nodes=new_nodes, edges=new_edges)

    batch_graphs = jraph.batch(unbatch)

    fully_graph, edge_weights = edge_weight_decoder.make_graph_fully_connected(batch_graphs, multi_edge_repeat)

    revert_graph = edge_weight_decoder.make_graph_sparse(fully_graph, edge_weights)

    for a, b in zip(jraph.unbatch(batch_graphs), jraph.unbatch(revert_graph)):
        assert jnp.allclose(b.nodes, a.nodes)
        assert jnp.sum(b.edges - a.edges) < 1e-12
        assert jnp.sum(b.senders - a.senders) < 1e-12
        assert jnp.sum(b.receivers - a.receivers) < 1e-12
        assert jnp.allclose(b.n_node, a.n_node)
        assert jnp.allclose(b.n_edge, a.n_edge)


@pytest.mark.parametrize("max_num_nodes, multi_edge_repeat, stack, mpg_stack", [(5, 1, [15, 76, 65, 1], [[2], [2, 4]]), (6, 2, [15, 7], [[11], [2, 3], [3]]), (10, 3, [2], [[2], [4], [5], [3], [2]])])
def test_edge_weight_decoder(max_num_nodes, multi_edge_repeat, stack, mpg_stack, jax_rng, ensure_tempfile, mlp_kwargs):
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

    model = edge_weight_decoder.EdgeWeightDecoder(max_nodes=max_num_nodes, init_node_stack=stack, init_edge_stack=stack, prob_mpg_stack=mpg_stack, multi_edge_repeat=multi_edge_repeat, mlp_kwargs=mlp_kwargs)

    rng, rng_normal = jax.random.split(rng)
    test_input = jax.random.normal(rng_normal, (graph_num, 7))
    test_input = jnp.hstack((test_input, n_node_edge))

    params = model.init({"params": init_rng, "dropout": dropout_rng}, test_input)
    rng, dropout_rng = jax.random.split(rng)

    apply_model = jax.jit(model.apply)
    apply_model(params, test_input, rngs={"dropout": dropout_rng})
    rng, dropout_rng = jax.random.split(rng)
    out_graphs, edge_weights = apply_model(params, test_input, rngs={"dropout": dropout_rng})

    out_graphs = edge_weight_decoder.make_graph_sparse(out_graphs, edge_weights)

    unbatch_nodes = jraph.unbatch(out_graphs)

    m = hashlib.shake_256()
    m.update(test_input.tobytes())

    original_path, _ = ensure_tempfile
    # metric_util.svg_graph_list(unbatch_nodes, filename=os.path.join(original_path, f"final_ew_decode{m.hexdigest(3)}.pdf"))

    for _i, graph in enumerate(unbatch_nodes):
        vals, count = jnp.unique(graph.senders, return_counts=True)
        assert jnp.min(vals) >= 0
        assert jnp.max(vals) <= graph.n_node[0] - 1

        vals, count = jnp.unique(graph.receivers, return_counts=True)
        assert jnp.min(vals) >= 0
        assert jnp.max(vals) <= graph.n_node[0] - 1

    rng, dropout_rng = jax.random.split(rng)

    def dummy_loss(params, graph):
        out_graph, edge_weights = model.apply(params, graph, rngs={"dropout": dropout_rng})
        return jnp.sum(out_graph.nodes) + jnp.sum(out_graph.edges) + jnp.sum(edge_weights)

    # Ensure we have gradients on all our params
    grads = jax.grad(dummy_loss)(params, test_input)
    for leaf in jax.tree_util.tree_leaves(grads):
        assert jnp.sum(jnp.abs(leaf)) > 0

    metric_model = mpg_edge_weight.MessagePassingEW(
        node_feature_sizes=mpg_stack,
        edge_feature_sizes=mpg_stack,
        global_feature_sizes=mpg_stack,
        mean_instead_of_sum=False,
    )
    rng, la_rng = jax.random.split(rng)
    metric_params = metric_model.init(la_rng, out_graphs)

    def global_loss(params, in_data):
        out_graph, edge_weights = model.apply(params, in_data, rngs={"dropout": dropout_rng})
        out_graph = out_graph._replace(globals=out_graph.globals * 0)

        sharp_loss = mpg_edge_weight.edge_weights_sharpness_loss(edge_weights)
        n_edge_loss = mpg_edge_weight.edge_weights_n_edge_loss(edge_weights, out_graph.n_edge[::2])

        metric_out = metric_model.apply(metric_params, out_graph, edge_weights)
        return jnp.mean(metric_out.globals) + sharp_loss + n_edge_loss

    func = jax.jit(jax.grad(global_loss))
    func = jax.grad(global_loss)
    func(params, test_input)
    grads = func(params, test_input)
    for leaf in jax.tree_util.tree_leaves(grads):
        assert jnp.sum(jnp.abs(leaf)) > 0
