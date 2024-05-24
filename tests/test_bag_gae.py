from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph
import optax
import pytest
from flax.training.train_state import TrainState

from gjp import bag_gae, mpg

MLP_KWARGS = {"dropout_rate": 0.1, "deterministic": False, "activation": nn.sigmoid}


def test_find_multi_edge_repeat(batch_graphs):
    multi_edge_repeat = bag_gae.find_multi_edge_repeat(batch_graphs)
    assert multi_edge_repeat == 3


def test_bag_gae(batch_graphs):
    max_num_nodes = jnp.max(batch_graphs.n_node) + 1
    multi_edge_repeat = bag_gae.find_multi_edge_repeat(batch_graphs)

    mlp_stack = [1, 4, 7, 2, 4]
    mpg_stack = [[2], [4, 8], [16], [2]]

    model = bag_gae.BagGAE(
        max_num_nodes=max_num_nodes,
        encoder_stack=mpg_stack,
        init_stack=mlp_stack,
        decoder_stack=mpg_stack,
        node_features=batch_graphs.nodes.shape[1],
        edge_features=batch_graphs.edges.shape[1],
        multi_edge_repeat=multi_edge_repeat,
        mlp_kwargs=MLP_KWARGS,
    )
    rng = jax.random.key(234)
    rng, rng_split_a, rng_split_b = jax.random.split(rng, 3)

    params = model.init({"params": rng, "reparametrize": rng_split_a, "dropout": rng_split_b}, batch_graphs)
    rng, rng_split_a, rng_split_b = jax.random.split(rng, 3)

    apply_model = jax.jit(lambda x, y, z, w: model.apply(x, y, rngs={"reparametrize": z, "dropout": w}))
    out_graphs = apply_model(params, batch_graphs, rng_split_a, rng_split_b)
    rng, rng_split = jax.random.split(rng)

    out_unbatch = jraph.unbatch(out_graphs)

    for i, in_graph in enumerate(jraph.unbatch(batch_graphs)):
        out_graph = out_unbatch[2 * i]
        assert out_graph.n_node == in_graph.n_node
        assert out_graph.n_edge[0] <= in_graph.n_edge[0]


@pytest.mark.parametrize("norm", [True, False])
def test_bag_loss_function(batch_graphs, norm):
    max_num_nodes = jnp.max(batch_graphs.n_node) + 1
    multi_edge_repeat = bag_gae.find_multi_edge_repeat(batch_graphs)

    mlp_stack = [1, 4, 7, 2, 4]
    mpg_stack = [[2], [4, 8], [16], [2]]

    model = bag_gae.BagGAE(
        max_num_nodes=max_num_nodes,
        encoder_stack=mpg_stack,
        init_stack=mlp_stack,
        decoder_stack=mpg_stack,
        node_features=batch_graphs.nodes.shape[1],
        edge_features=batch_graphs.edges.shape[1],
        multi_edge_repeat=multi_edge_repeat,
        mlp_kwargs=MLP_KWARGS,
    )
    rng = jax.random.key(234)
    rng, rng_split_a, rng_split_b = jax.random.split(rng, 3)

    params = model.init({"params": rng, "reparametrize": rng_split_a, "dropout": rng_split_b}, batch_graphs)
    rng, rng_split_a, rng_split_b = jax.random.split(rng, 3)
    tx = optax.adam(learning_rate=1e-3)
    opt_state = tx.init(params)
    train_state = TrainState(step=0, apply_fn=model.apply, params=params, tx=tx, opt_state=opt_state)
    train_state.apply_fn(train_state.params, batch_graphs, rngs={"reparametrize": rng_split_a, "dropout": rng_split_b})

    metric_model = mpg.MessagePassingGraph(
        node_stack=mpg_stack,
        edge_stack=mpg_stack,
        attention_stack=mpg_stack,
        global_stack=mpg_stack,
    )
    rng, rng_split = jax.random.split(rng)
    metric_params = metric_model.init(rng_split, batch_graphs)
    metric_state = TrainState(apply_fn=metric_model.apply, params=metric_params, step=None, tx=None, opt_state=None)

    partial_step = partial(bag_gae.train_step, metric_state=metric_state, norm=norm)
    func = jax.jit(partial_step)

    for _ in range(3):
        train_state, train_loss, train_loss_a, train_loss_b, test_loss_a, test_loss_b = func(batch_graphs, batch_graphs, train_state, rng=rng_split)
        rng, rng_split = jax.random.split(rng)

        print(train_loss, train_loss_a, train_loss_b, test_loss_a, test_loss_b)
