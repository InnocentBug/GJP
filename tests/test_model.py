import jax
import jax.numpy as jnp
import jraph
import pytest

from gjp import MessagePassing, model


def test_simple_pass(batch_graphs):
    final_node_size = 21
    final_edge_size = 34
    final_global_size = 42

    node_stack = [[4], [2], [final_node_size]]
    edge_stack = [[4], [2], [final_edge_size]]
    global_stack = [[4], [2], [final_global_size]]

    rng = jax.random.key(42)
    rng, init_rng = jax.random.split(rng)
    model = MessagePassing(node_stack, edge_stack, global_stack)
    params = model.init(init_rng, batch_graphs)

    def apply_model(x):
        return model.apply(params, x)

    out_graph = apply_model(batch_graphs)
    assert out_graph.nodes.shape[1] == final_node_size
    assert out_graph.edges.shape[1] == final_edge_size
    assert out_graph.globals.shape[1] == final_global_size


@pytest.mark.parametrize("mean_instead_of_sum", (True, False))
def test_mean_instead_of_sum(batch_graphs, mean_instead_of_sum):
    final_node_size = 21
    final_edge_size = 34
    final_global_size = 42

    node_stack = [[4], [2], [final_node_size]]
    edge_stack = [[4], [2], [final_edge_size]]
    global_stack = [[4], [2], [final_global_size]]

    rng = jax.random.key(42)
    rng, init_rng = jax.random.split(rng)
    model = MessagePassing(node_stack, edge_stack, global_stack, mean_instead_of_sum=mean_instead_of_sum)
    params = model.init(init_rng, batch_graphs)

    def apply_model(x):
        return model.apply(params, x)

    out_graph = apply_model(batch_graphs)
    assert out_graph.nodes.shape[1] == final_node_size
    assert out_graph.edges.shape[1] == final_edge_size
    assert out_graph.globals.shape[1] == final_global_size


@pytest.mark.parametrize(
    "edge_stack, node_stack, global_stack",
    [
        ([[2], [3, 4]], [[2], [3, 4]], [None, [3, 4]]),
        (None, [[2], [3, 4]], [[2], [3, 4]]),
        ([[2], None], None, [[2], [3, 4]]),
        ([[2], [3, 4]], [[2], [3, 4]], None),
        ([[2], None], None, None),
        (None, [None, [3, 4]], None),
        (None, None, [[2], [3, 4]]),
        (None, None, None),
        ([None], [None], [None]),
    ],
)
def test_stacks(batch_graphs, edge_stack, node_stack, global_stack):
    final_node_size = 21
    final_edge_size = 34
    final_global_size = 42

    if edge_stack:
        edge_stack += [[final_edge_size]]
    if node_stack:
        node_stack += [[final_node_size]]
    if global_stack:
        global_stack += [[final_global_size]]

    rng = jax.random.key(42)
    rng, init_rng = jax.random.split(rng)
    model = MessagePassing(node_stack, edge_stack, global_stack)
    params = model.init(init_rng, batch_graphs)

    def apply_model(x):
        return model.apply(params, x)

    out_graph = apply_model(batch_graphs)
    if node_stack:
        assert out_graph.nodes.shape[1] == final_node_size
    else:
        assert jnp.allclose(out_graph.nodes, batch_graphs.nodes)
    if edge_stack:
        assert out_graph.edges.shape[1] == final_edge_size
    else:
        assert jnp.allclose(out_graph.edges, batch_graphs.edges)
    if global_stack:
        assert out_graph.globals.shape[1] == final_global_size
    else:
        jnp.allclose(out_graph.globals, batch_graphs.globals)


def test_batching(batch_graphs):
    final_node_size = 21
    final_edge_size = 34
    final_global_size = 4

    node_stack = [[4], [2], [final_node_size]]
    edge_stack = [[4], [2], [final_edge_size]]
    global_stack = [[4], [2], [final_global_size]]

    rng = jax.random.key(42)
    rng, init_rng = jax.random.split(rng)
    model = MessagePassing(node_stack, edge_stack, global_stack)
    params = model.init(init_rng, batch_graphs)

    batched_out = model.apply(params, batch_graphs)
    unbatch_out = jraph.unbatch(batched_out)

    rtol = 1e-4
    atol = 1e-7
    for i, in_graph in enumerate(jraph.unbatch(batch_graphs)):
        graph = model.apply(params, in_graph)
        batch_graph = unbatch_out[i]

        assert jnp.allclose(batch_graph.nodes, graph.nodes, rtol=rtol, atol=atol)
        assert jnp.allclose(batch_graph.edges, graph.edges, rtol=rtol, atol=atol)
        assert jnp.allclose(batch_graph.senders, graph.senders)
        assert jnp.allclose(batch_graph.receivers, graph.receivers)
        assert jnp.allclose(batch_graph.n_node, graph.n_node)
        assert jnp.allclose(batch_graph.n_edge, graph.n_edge)

        global_a = batch_graph.globals
        global_b = graph.globals
        assert jnp.sqrt(jnp.sum((global_a - global_b) ** 2)) < 5e-6
        assert jnp.allclose(global_a, global_b, rtol=rtol, atol=atol)


def test_split_and_sum():
    input_array = jnp.asarray([[1, 1], [2, 1], [3, 3], [4, 4], [56, 6], [8, 8], [9, 8], [9, 8], [0, 1]])
    input_index = jnp.asarray([2, 4, 2, 1])

    output = model.split_and_sum(input_array, input_index)
    assert jnp.allclose(output, jnp.asarray([[3, 2], [71, 21], [18, 16], [0, 1]]))

    input_array2 = jnp.asarray([[1, 1], [2, 1], [3, 3], [4, 4], [56, 6], [8, 8], [9, 8], [9, 8]])
    input_index2 = jnp.asarray([0, 2, 4, 0, 2, 0])
    output2 = model.split_and_sum(input_array2, input_index2)
    assert jnp.allclose(output2, jnp.asarray([[0, 0], [3, 2], [71, 21], [0, 0], [18, 16], [0, 0]]))

    input_array3 = jnp.asarray(jnp.ones((0, 5)))
    input_index3 = jnp.asarray([0])
    output3 = model.split_and_sum(input_array3, input_index3)
    assert output3.shape == (1, 5)


def test_split_and_mean():
    input_array = jnp.asarray([[1, 1], [2, 1], [3, 3], [4, 4], [56, 6], [8, 8], [9, 8], [9, 8], [0, 1]])
    input_index = jnp.asarray([2, 4, 2, 1])

    output = model.split_and_mean(input_array, input_index)
    assert jnp.allclose(output, jnp.asarray([[1.5, 1], [17.75, 5.25], [9, 8], [0, 1]]))

    input_array2 = jnp.asarray([[1, 1], [2, 1], [3, 3], [4, 4], [56, 6], [8, 8], [9, 8], [9, 8]])
    input_index2 = jnp.asarray([0, 2, 4, 0, 2, 0])
    output2 = model.split_and_mean(input_array2, input_index2)
    assert jnp.allclose(output2, jnp.asarray([[0, 0], [1.5, 1], [17.75, 5.25], [0, 0], [9, 8], [0, 0]]))

    input_array3 = jnp.asarray(jnp.ones((0, 5)))
    input_index3 = jnp.asarray([0])
    output3 = model.split_and_mean(input_array3, input_index3)
    assert output3.shape == (1, 5)
