import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph
import optax
import pytest
from flax.training.train_state import TrainState
from utils import compare_graphs

from gjp import MLP, bag_gae, edge_weight_decoder, mpg_edge_weight


@pytest.mark.parametrize("final_size", [1, 10, 100])
def test_train_edge_weights(jax_rng, final_size):

    def train_step(data, n_edge, state):
        def loss_function(params, data):
            edge_weights = state.apply_fn(params, data)
            loss = mpg_edge_weight.edge_weights_sharpness_loss(edge_weights)
            loss += 1 * mpg_edge_weight.edge_weights_n_edge_loss(edge_weights, n_edge)
            return loss

        val, grads = jax.value_and_grad(loss_function)(state.params, data)
        state = state.apply_gradients(grads=grads)
        return state, val

    class Model(nn.Module):
        final_size: int

        @nn.compact
        def __call__(self, x):
            mlp_a = MLP(
                [
                    32,
                    64,
                    32,
                ]
            )
            mlp_b = MLP([self.final_size], activation=nn.sigmoid)

            def func(x):
                x = mlp_a(x)
                x = mlp_b(x)
                return x

            x = jax.vmap(func)(x)
            return x

    num_array = 4
    test_input = jax.random.normal(jax_rng, (num_array, 2))

    n_edge = jax.random.randint(jax_rng, (num_array,), 0, final_size, dtype=int)
    model = Model(final_size)
    params = model.init(jax_rng, test_input)
    val_a = model.apply(params, test_input)
    assert mpg_edge_weight.edge_weights_sharpness_loss(val_a) > 0.1
    assert mpg_edge_weight.edge_weights_n_edge_loss(val_a, n_edge) > 0.1

    tx = optax.adamw(learning_rate=1e-3)
    opt_state = tx.init(params)
    state = TrainState(params=params, apply_fn=model.apply, tx=tx, opt_state=opt_state, step=0)

    jit_step = jax.jit(train_step)
    last_loss = None
    for i in range(50000):
        state, val = jit_step(test_input, n_edge, state)
        if i % 1000 == 0:
            if last_loss is not None:
                assert val <= 10 * last_loss
            if last_loss is None or val < last_loss:
                last_loss = val

    val_b = model.apply(state.params, test_input)
    assert mpg_edge_weight.edge_weights_sharpness_loss(val_b) < 1e-6
    assert mpg_edge_weight.edge_weights_n_edge_loss(val_b, n_edge) < 1e-6
    for i, n in enumerate(n_edge):
        assert jnp.abs(jnp.sum(val_b[i]) - n) < 0.1


@pytest.mark.parametrize("node_stack, edge_stack, global_stack, mean_sum", [(None, None, None, True), ([2, 3, 5], [3, 4], [12, 54, 2], True), (None, None, None, False), ([2, 3, 5], [3, 4], [12, 54, 2], False)])
def test_ew_message_passing_layer(batch_graphs, node_stack, edge_stack, global_stack, mean_sum):

    rng = jax.random.key(321)
    node_end = None
    if node_stack is not None:
        node_end = node_stack[-1]
    edge_end = None
    if edge_stack:
        edge_end = edge_stack[-1]

    model = mpg_edge_weight.MessagePassingLayerEW(node_feature_sizes=node_stack, edge_feature_sizes=edge_stack, global_feature_sizes=global_stack, mean_instead_of_sum=mean_sum, mlp_kwargs={"dropout_rate": 0.1})
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
    a = jnp.hstack(jax.tree.flatten(jax.tree.map(jnp.ravel, un_graph))[0])
    b = jnp.hstack(jax.tree.flatten(jax.tree.map(jnp.ravel, unbatch2[graph_idx]))[0])
    assert jnp.allclose(a, b)

    weights = jnp.ones(jnp.sum(batch_graphs.n_edge))
    out_graphs2 = apply_model(params, batch_graphs, weights)

    assert jnp.allclose(out_graphs2.globals, out_graphs.globals)
    assert jnp.allclose(out_graphs2.nodes, out_graphs.nodes)
    assert jnp.allclose(out_graphs2.edges, out_graphs.edges)
    assert jnp.allclose(out_graphs2.senders, out_graphs.senders)
    assert jnp.allclose(out_graphs2.receivers, out_graphs.receivers)

    # print(batch_graphs.globals)
    # print(out_graphs.globals)
    # print(out_graphs2.globals)


@pytest.mark.parametrize("mpg_stack", [[None, [1, 2], [3, 4]], None, [[2], [3], [4]]])
def test_edge_weight_metric(jax_rng, batch_graphs, mpg_stack):
    multi_edge_repeat = bag_gae.find_multi_edge_repeat(batch_graphs)
    fully_graph, edge_weights = edge_weight_decoder.make_graph_fully_connected(batch_graphs, multi_edge_repeat)

    model = mpg_edge_weight.MessagePassingEW(node_feature_sizes=mpg_stack, edge_feature_sizes=mpg_stack, global_feature_sizes=mpg_stack)

    params = model.init(jax_rng, batch_graphs)

    graph_a = model.apply(params, batch_graphs)
    graph_b = model.apply(params, fully_graph, edge_weights)

    result_a = graph_a.globals
    result_b = graph_b.globals[::2]
    assert result_a.shape == result_b.shape
    assert jnp.allclose(result_a, result_b)


def test_small(jax_rng):
    stack = [[16, 2], [6, 1]]
    graph = jraph.GraphsTuple(
        nodes=jnp.asarray([1, 2, 3]).reshape((3, 1)),
        edges=jnp.asarray([5, 6]).reshape((2, 1)),
        senders=jnp.asarray([0, 1], dtype=int),
        receivers=jnp.asarray([2, 0], dtype=int),
        n_node=jnp.asarray([3], dtype=int),
        n_edge=jnp.asarray([2], dtype=int),
        globals=jnp.asarray([[0.0]]),
    )

    fully_graph, edge_weights = edge_weight_decoder.make_graph_fully_connected(graph, 1)

    model = mpg_edge_weight.MessagePassingEW(
        node_feature_sizes=stack,
        edge_feature_sizes=stack,
        global_feature_sizes=stack,
    )

    params = model.init(jax_rng, graph)

    result_a = model.apply(params, graph)
    result_b = model.apply(params, fully_graph, edge_weights)

    revert_graph = edge_weight_decoder.make_graph_sparse(result_b, edge_weights)

    assert jnp.allclose(result_a.nodes, revert_graph.nodes)
    assert jnp.abs(jnp.sum(result_a.edges) - jnp.sum(revert_graph.edges)) < 1e-7
    assert jnp.allclose(result_a.n_node, revert_graph.n_node)
    assert jnp.abs(jnp.sum(result_a.senders) - jnp.sum(revert_graph.senders)) < 1e-10
    assert jnp.abs(jnp.sum(result_a.receivers) - jnp.sum(revert_graph.receivers)) < 1e-10

    assert jnp.allclose(result_a.globals[0], result_b.globals[0])
