import jax
import jax.numpy as jnp
import jraph
import pytest

from gjp import bag_gae, edge_weight_decoder


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
    max_nodes = jnp.max(batch_graphs.n_node)

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
