from functools import partial

import jax
import jax.numpy as jnp
import jraph
import optax
import pytest
from utils import compare_graphs

from gjp import metric_util, mpg


@pytest.mark.parametrize("node_stack, edge_stack, attention_stack, global_stack", [(None, None, None, None), ([2, 3, 5], [3, 4], [1, 12, 2, 4], [12, 54, 2])])
def test_message_passing_layer(batch_graphs, node_stack, edge_stack, attention_stack, global_stack):
    rng = jax.random.key(321)
    node_end = None
    if node_stack is not None:
        node_end = node_stack[-1]
    edge_end = None
    if edge_stack:
        edge_end = edge_stack[-1]

    model = mpg.MessagePassingGraphLayer(node_stack=node_stack, edge_stack=edge_stack, attention_stack=attention_stack, mlp_kwargs={"dropout_rate": 0.1})
    params = model.init(rng, batch_graphs)

    model = mpg.MessagePassingGraphLayer(
        node_stack=node_stack,
        edge_stack=edge_stack,
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


@pytest.mark.parametrize("mean_aggregate", (True, False))
def test_simple_pass(batch_graphs, mean_aggregate):
    final_node_size = 21
    final_edge_size = 34
    final_global_size = 42

    node_stack = [[4], [2], [final_node_size]]
    edge_stack = [[4], [2], [final_edge_size]]
    global_stack = [[4], [2], [final_global_size]]
    attention_stack = [[4], [2], [7]]

    rng = jax.random.key(42)
    rng, init_rng = jax.random.split(rng)
    model = mpg.MessagePassingGraph(node_stack, edge_stack, attention_stack, global_stack, mean_aggregate)
    params = model.init(init_rng, batch_graphs)

    apply_model = jax.jit(lambda x: model.apply(params, x))
    out_graph = apply_model(batch_graphs)
    assert out_graph.nodes.shape[1] == final_node_size
    assert out_graph.edges.shape[1] == final_edge_size
    assert out_graph.globals.shape[1] == final_global_size


@pytest.mark.parametrize(
    "edge_stack, node_stack, attention_stack, global_stack",
    [
        ([[2], [3, 4]], [[2], [3, 4]], [[2], [2, 3]], [None, [3, 4]]),
        (None, [[2], [3, 4]], [[2], [3]], [[2], [3, 4]]),
        (
            [[2], None],
            None,
            [
                [
                    2,
                ],
                [3],
            ],
            [[2], [3, 4]],
        ),
        ([[2], [3, 4]], [[2], [3, 4]], [[2], [3]], None),
        ([[2], [3, 4]], None, None, [[2], [3, 4]]),
        ([[2], None], None, None, None),
        (None, [None, [3, 4]], [None, [2]], None),
        (None, None, None, [[2], [3, 4]]),
        (None, None, None, None),
        ([None], [None], [None], [None]),
    ],
)
def test_stacks(batch_graphs, edge_stack, node_stack, attention_stack, global_stack):
    final_node_size = 21
    final_edge_size = 34
    final_global_size = 42

    if edge_stack:
        edge_stack += [[final_edge_size]]
    if node_stack:
        node_stack += [[final_node_size]]
    if global_stack:
        global_stack += [[final_global_size]]
    if attention_stack:
        attention_stack += [[11]]

    rng = jax.random.key(42)
    rng, init_rng = jax.random.split(rng)
    model = mpg.MessagePassingGraph(node_stack, edge_stack, attention_stack, global_stack)
    params = model.init(init_rng, batch_graphs)

    apply_model = jax.jit(lambda x: model.apply(params, x))
    _ = apply_model(batch_graphs)
    out_graph = apply_model(batch_graphs)

    if node_stack:
        assert out_graph.nodes.shape[1] == final_node_size
    else:
        assert jnp.allclose(out_graph.nodes, batch_graphs.nodes)
    if edge_stack:
        assert out_graph.edges.shape[1] == final_edge_size
    else:
        if attention_stack is None:
            assert jnp.allclose(out_graph.edges, batch_graphs.edges)
        else:
            assert not jnp.allclose(out_graph.edges, batch_graphs.edges)
            assert out_graph.edges.shape == batch_graphs.edges.shape
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
    attention_stack = [[4], [2], [11]]

    rng = jax.random.key(42)
    rng, init_rng = jax.random.split(rng)
    model = mpg.MessagePassingGraph(node_stack, edge_stack, attention_stack, global_stack)
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


def train_step(batch_train, batch_test, opt_state, params, i, model, tx):
    loss_fn = partial(metric_util.loss_function_combined, model=model, norm=True, norm_step=i / 1e5)
    loss_fn_wo_norm = partial(metric_util.loss_function_combined, model=model, norm=False)
    loss_grad_fn = jax.value_and_grad(loss_fn)

    train_loss, grads = loss_grad_fn(params, batch_train)
    test_loss = loss_fn_wo_norm(params, batch_test)
    train_loss2 = loss_fn_wo_norm(params, batch_train)

    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return params, opt_state, train_loss, train_loss2, test_loss


def test_learning(batch_graphs):
    final_node_size = 7
    final_edge_size = 4
    final_global_size = 2

    node_stack = [[4], [2], [final_node_size]]
    edge_stack = [[4], [2], [final_edge_size]]
    global_stack = [[4], [2], [final_global_size]]
    attention_stack = [[4], [2], [7]]

    rng = jax.random.key(42)
    rng, init_rng = jax.random.split(rng)
    model = mpg.MessagePassingGraph(node_stack, edge_stack, attention_stack, global_stack)
    params = model.init(init_rng, batch_graphs)

    tx = optax.adam(learning_rate=1e-5)
    opt_state = tx.init(params)
    partial_step = partial(train_step, model=model, tx=tx)
    jit_step = jax.jit(partial_step)

    for i in range(10):
        params, opt_state, train_loss, train_loss2, test_loss = jit_step(batch_graphs, batch_graphs, opt_state, params, i)
        print(i, train_loss, train_loss2, test_loss)

    out_graph = model.apply(params, batch_graphs)
    print(out_graph)
