import jax.numpy as jnp


def compare_graphs(a, b, node_end, edge_end):
    assert jnp.allclose(a.senders, b.senders)
    assert jnp.allclose(a.receivers, b.receivers)
    assert jnp.allclose(a.n_edge, b.n_edge)
    assert jnp.allclose(a.n_node, b.n_node)

    if node_end:
        assert a.nodes.shape == b.nodes.shape[:-1] + (node_end,)
    if edge_end:
        assert a.edges.shape == b.edges.shape[:-1] + (edge_end,)
