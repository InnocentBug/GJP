import random

import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph
import optax
import pytest
from flax.training.train_state import TrainState
from utils import compare_graphs

from gjp import MLP, bag_gae, edge_weight_decoder, mpg_edge_weight


@pytest.mark.parametrize("seed", [11, 12, 42])
def test_gumbel_topk_val(seed):
    n_batch = 5
    n_size = 4
    temperature = 1
    rng = jax.random.key(seed)

    x = jax.random.normal(rng, (n_batch, n_size))
    n_edge = jnp.asarray([0.0, 1, 2.0, 3, 4.0])

    edge_weights = mpg_edge_weight.gumbel_topk(x, n_edge, n_size * 2 + 2, rng, temperature, pre_filter=jax.nn.sigmoid)
    lossA = mpg_edge_weight.edge_weights_sharpness_loss(edge_weights)
    lossB = mpg_edge_weight.edge_weights_n_edge_loss(edge_weights, n_edge)

    print(lossA, lossB)
    # print(edge_weights)

    assert lossA < 0.1
    assert lossB < 0.1


@pytest.mark.parametrize("seed, temperature", [(145, 1), (234, 0.75), (34, 0.5), (541, 0.1)])
def test_gumbel_topk_grad(seed, temperature):
    n_batch = 5
    n_size = 4
    rng = jax.random.key(seed)

    rng, use_rng = jax.random.split(rng)
    x = jax.random.normal(use_rng, (n_batch, n_size))
    rng, use_rng = jax.random.split(rng)
    y = jax.random.normal(use_rng, (n_batch, n_size))

    # Gradients for 0 are going to be zero
    n_edge = jnp.asarray([1, 2.0, 3, 2, 4.0])

    def loss_function(x, y, rng):
        edge_weights = mpg_edge_weight.gumbel_topk(x, n_edge, n_size + 2, rng, temperature, pre_filter=jax.nn.sigmoid)
        lossA = mpg_edge_weight.edge_weights_sharpness_loss(edge_weights)
        lossB = mpg_edge_weight.edge_weights_n_edge_loss(edge_weights, n_edge)
        lossC = jnp.mean(y * edge_weights)
        return lossA + lossB + lossC

    jit_loss = jax.jit(jax.value_and_grad(loss_function))

    for _i in range(5):
        rng, use_rng = jax.random.split(rng)
        val, grad = jit_loss(x, y, use_rng)
        print(grad)
        non_zero = jnp.nonzero(grad)[0].size
        # Heuristics to make sure our gradients aren't 0 in too many places
        assert grad.size - non_zero < 1 / temperature + 4


@pytest.mark.parametrize("final_size, temperature", [(4, 0.5), (20, 0.1), (100, 0.25)])
def test_train_edge_weights(jax_rng, final_size, temperature):
    def train_step(data, n_edge, state, val_x, val_y, temperature, gumbel_rng):
        def loss_function(params, data_x, data_y):
            edge_weights = state.apply_fn(params, data_x, temperature, rngs={"gumbel": gumbel_rng})

            lossA = mpg_edge_weight.edge_weights_sharpness_loss(edge_weights)
            lossB = mpg_edge_weight.edge_weights_n_edge_loss(edge_weights, data_y)
            return lossA + lossB

        val, grads = jax.value_and_grad(loss_function)(state.params, data, n_edge)

        val_weights = state.apply_fn(params, val_x, temperature, rngs={"gumbel": gumbel_rng})
        lossA = mpg_edge_weight.edge_weights_sharpness_loss(val_weights)
        lossB = mpg_edge_weight.edge_weights_n_edge_loss(val_weights, val_y)

        state = state.apply_gradients(grads=grads)
        return state, val, lossA, lossB

    class Model(nn.Module):
        final_size: int

        @nn.compact
        def __call__(self, x, temperature):
            n_edge = x[:, -1]
            mlp_a = MLP([32, 64, 64, 32, final_size])

            def func(x):
                x = mlp_a(x)
                return x

            x = jax.vmap(func)(x)
            rng = self.make_rng("gumbel")
            x = mpg_edge_weight.gumbel_topk(x, n_edge, final_size + 10, rng, temperature, pre_filter=jax.nn.sigmoid)

            # x = (jnp.tanh( (x-0.5)*2*jnp.pi) + 1)/2
            # jax.debug.print("x {}", x)
            # jax.debug.print("n_edge {}", n_edge)
            return x

    num_array = 400
    rng, jax_rng = jax.random.split(jax_rng)
    test_input = jax.random.normal(jax_rng, (num_array, 2))

    n_edge = jax.random.randint(jax_rng, (num_array, 1), 0, final_size, dtype=int)
    test_input = jnp.hstack([test_input, n_edge])

    test_input = n_edge

    train_frac = 0.9
    train_x = test_input[: int(num_array * train_frac)]
    test_x = test_input[int(num_array * train_frac) :]
    train_y = n_edge[: int(num_array * train_frac)]
    test_y = n_edge[int(num_array * train_frac) :]

    batch_frac = 0.1
    batch_x = []
    batch_y = []

    start = 0
    end = int(num_array * train_frac * batch_frac)
    while end < int(num_array * train_frac):
        batch_x += [train_x[start:end]]
        batch_y += [train_y[start:end]]

        start = end
        end += int(num_array * train_frac * batch_frac)

    model = Model(final_size)
    rng, init, gumbel = jax.random.split(jax_rng, 3)

    params = model.init({"params": init, "gumbel": gumbel}, test_input, temperature)

    # rng, gumbel = jax.random.split(rng)
    # val_a = model.apply(params, test_x, temperature=temperature, rngs={"gumbel": gumbel})
    # assert mpg_edge_weight.edge_weights_sharpness_loss(val_a) > 0.1
    # assert mpg_edge_weight.edge_weights_n_edge_loss(val_a, n_edge) > 0.1

    tx = optax.adamw(learning_rate=1e-3)
    opt_state = tx.init(params)
    state = TrainState(params=params, apply_fn=model.apply, tx=tx, opt_state=opt_state, step=0)

    jit_step = jax.jit(train_step)

    for i in range(100):
        perm = list(range(len(batch_x)))
        random.shuffle(perm)
        for j in perm:
            rng, gumbel = jax.random.split(rng)
            state, loss, val_lossA, val_lossB = jit_step(batch_x[j], batch_y[j], state, test_x, test_y, temperature, gumbel)
        if i % 10 == 0:
            print(i, loss, val_lossA, val_lossB)

    rng, gumbel = jax.random.split(rng)
    val_b = model.apply(state.params, test_x, temperature, rngs={"gumbel": gumbel})
    lossA = mpg_edge_weight.edge_weights_sharpness_loss(val_b)
    lossB = mpg_edge_weight.edge_weights_n_edge_loss(val_b, test_y)
    print(lossA, lossB)

    assert lossA < 0.1
    assert lossB < 0.1
    for i, n in enumerate(test_y):

        assert jnp.abs(jnp.sum(val_b[i]) - n) < 1.1


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
