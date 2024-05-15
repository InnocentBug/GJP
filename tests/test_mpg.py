import jax
import jax.numpy as jnp
import jraph
import pytest

from gjp import mpg


def compare_graphs(a, b, node_end, edge_end):
    assert jnp.allclose(a.senders, b.senders)
    assert jnp.allclose(a.receivers, b.receivers)
    assert jnp.allclose(a.n_edge, b.n_edge)
    assert jnp.allclose(a.n_node, b.n_node)

    if node_end:
        assert a.nodes.shape == b.nodes.shape[:-1] + (node_end,)
    if edge_end:
        assert a.edges.shape == b.edges.shape[:-1] + (edge_end,)


@pytest.mark.parametrize("node_stack, edge_stack, attention_stack, global_stack", [(None, None, None, None), ([2, 3, 5], [3, 4], [1, 12, 2, 4], [12, 54, 2])])
def test_message_passing_layer(batch_graphs, node_stack, edge_stack, attention_stack, global_stack):
    rng = jax.random.key(321)
    node_end = None
    if node_stack is not None:
        node_end = node_stack[-1]
    edge_end = None
    if edge_stack:
        edge_end = edge_stack[-1]

    model = mpg.MessagePassingGraphLayer(node_feature_size=node_stack, edge_feature_size=edge_stack, attention_stack=attention_stack, mlp_kwargs={"dropout_rate": 0.1})
    params = model.init(rng, batch_graphs)

    model = mpg.MessagePassingGraphLayer(
        node_feature_size=node_stack,
        edge_feature_size=edge_stack,
        attention_stack=attention_stack,
        global_stack=global_stack,
    )
    params = model.init(rng, batch_graphs)

    apply_model = jax.jit(model.apply)
    apply_model(params, batch_graphs)
    out_graphs = apply_model(params, batch_graphs)
    compare_graphs(out_graphs, batch_graphs, node_end, edge_end)

    graph_idx = -2
    unbatch = jraph.unbatch(batch_graphs)
    un_graph = model.apply(params, unbatch[graph_idx])

    compare_graphs(un_graph, unbatch[graph_idx], node_end, edge_end)

    # Check that result of the model is the same, batched or unbatched
    unbatch2 = jraph.unbatch(out_graphs)
    a = jnp.hstack(jax.tree_flatten(jax.tree_map(jnp.ravel, un_graph))[0])
    b = jnp.hstack(jax.tree_flatten(jax.tree_map(jnp.ravel, unbatch2[graph_idx]))[0])
    assert jnp.allclose(a, b)


# def test_simple_pass():
#     with GraphData(".test_simple_pass") as dataset:
#         train, test = dataset.get_test_train(10, 5, 5, 11)
#         max_node_pad = 32
#         max_edges_pad = 64

#         train_jraph = convert_to_jraph(train, max_node_pad, max_edges_pad)
#         test_jraph = convert_to_jraph(test, max_node_pad, max_edges_pad)

#         dataset.get_similar_feature_graphs(train_jraph[0], 20)
#         train += [train[0]] * 20
#         dataset.get_similar_feature_graphs(test_jraph[0], 10)
#         test += [test[0]] * 10

#         final_node_size = 21
#         final_edge_size = 34
#         final_global_size = 42

#         node_stack = [[4], [2], [final_node_size]]
#         edge_stack = [[4], [2], [final_edge_size]]
#         global_stack = [[4], [2], [final_global_size]]

#         rng = jax.random.key(42)
#         rng, init_rng = jax.random.split(rng)
#         model = MessagePassing(node_stack, edge_stack, global_stack, num_nodes=max_node_pad)
#         params = model.init(init_rng, train_jraph[0])

#         apply_model = jax.jit(lambda x: model.apply(params, x))
#         out_graph = apply_model(test_jraph[0])
#         assert out_graph.nodes.shape[1] == final_node_size
#         assert out_graph.edges.shape[1] == final_edge_size
#         assert out_graph.globals.shape[1] == final_global_size
