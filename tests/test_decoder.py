import jax
import jax.numpy as jnp
import jraph
import pytest

from gjp import decoder

jax.config.update("jax_platform_name", "cpu")


# @pytest.mark.parametrize("max_num_nodes, max_num_edges, n_edge_features", [(3,6, 3), (7,3,5), (50,150,7)])
# def test_initial_edge_decoder(max_num_nodes, max_num_edges, n_edge_features):
#     model = decoder.InitialEdgeDecoder(mlp_stack=[16, 32, 64], max_num_nodes=max_num_nodes, max_num_edges=max_num_edges, n_edge_features=n_edge_features)

#     test_input = jnp.asarray([[1, 5, 8, 6.1, 44], [1, 5, 8, 6.1, 4.4]])

#     rng = jax.random.key(15)
#     rng, init_rng = jax.random.split(rng)
#     params = model.init(init_rng, test_input)

#     apply_model = jax.jit(lambda x: model.apply(params, x))

#     senders, receivers, features = apply_model(test_input)

#     assert senders.shape == (test_input.shape[0], max_num_edges)
#     assert receivers.shape == (test_input.shape[0], max_num_edges)

#     for i in range(test_input.shape[0]):
#         assert jnp.min(senders[i]) >= i*max_num_nodes
#         assert jnp.max(senders[i]) <= (i+1) * max_num_nodes
#         assert jnp.min(receivers[i]) >= i*max_num_nodes
#         assert jnp.max(receivers[i]) <= (i+1) * max_num_nodes

#     assert features.shape == (2, max_num_edges, n_edge_features)

# @pytest.mark.parametrize("max_num_nodes, n_node_features", [(3, 1), (6,4), (150, 16)])
# def test_initial_node_decoder(max_num_nodes, n_node_features):
#     model = decoder.InitialNodeDecoder(mlp_stack=[15, 76, 651], max_num_nodes=max_num_nodes, n_node_features=n_node_features)

#     test_input = jnp.asarray([[1, 5, 8, 6.1, 44], [1, 5, 8, 6.1, 44]])
#     rng = jax.random.key(15)
#     rng, init_rng = jax.random.split(rng)
#     params = model.init(init_rng, test_input)

#     apply_model = jax.jit(lambda x: model.apply(params, x))
#     initial_node_features = apply_model(test_input)
#     assert initial_node_features.shape == (test_input.shape[0], max_num_nodes, n_node_features)


@pytest.mark.parametrize("init_edge_features, init_node_features, max_num_nodes, max_num_edges", [(3, 2, 4, 8), (7, 9, 50, 100)])
def test_initial_graph_decoder(init_edge_features, init_node_features, max_num_nodes, max_num_edges):

    model = decoder.InitGraphDecoder(init_edge_stack=[2, 4, 8], init_edge_features=init_edge_features, init_node_stack=[5, 10, 7], init_node_features=init_node_features, max_num_nodes=max_num_nodes, max_num_edges=max_num_edges)

    test_input = jnp.asarray([[0.1, 0.2, 0.3, 5, 6], [0.5, 0.2, 0.4, 7, 14]])

    rng = jax.random.key(15)
    rng, init_rng = jax.random.split(rng)
    params = model.init(init_rng, test_input)

    out_graph = model.apply(params, test_input)
    out_batch = jraph.unbatch(out_graph)
    assert len(out_batch) == test_input.shape[0]
    for i, graph in enumerate(out_batch):
        assert graph.nodes.shape == (max_num_nodes, init_node_features)
        assert graph.edges.shape == (max_num_edges, init_edge_features)

        assert graph.senders.shape == (max_num_edges,)
        assert jnp.min(graph.senders) >= 0
        assert jnp.max(graph.senders) <= max_num_nodes

        assert graph.receivers.shape == (max_num_edges,)
        assert jnp.min(graph.receivers) >= 0
        assert jnp.max(graph.receivers) <= max_num_nodes

        assert jnp.allclose(graph.globals, test_input[i])
