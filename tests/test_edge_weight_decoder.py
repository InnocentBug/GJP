import jax
import jax.numpy as jnp
import jraph
import pytest

from gjp import edge_weight_decoder


@pytest.mark.parametrize("max_num_nodes, multi_edge_repeat", [(5, 1), (6, 2), (10, 3)])
def test_full_edge_decoder(max_num_nodes, multi_edge_repeat, jax_rng):
    graph_num = 3

    n_node_edge = jax.random.randint(jax_rng, (graph_num, 2), 2, max_num_nodes - 1)

    rng, rng_normal = jax.random.split(jax_rng)
    test_input = jax.random.normal(rng_normal, (graph_num, 7))
    test_input = jnp.hstack((test_input, n_node_edge))

    model = edge_weight_decoder.FullyConnectedGraph(max_num_nodes, multi_edge_repeat)
    params = model.init(rng, test_input)

    apply_model = jax.jit(model.apply)
    apply_model(params, test_input)
    rng, dropout_rng = jax.random.split(rng)
    out_graph = apply_model(params, test_input, rngs={"dropout": dropout_rng})

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
