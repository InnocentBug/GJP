from functools import partial

import jax
import jax.numpy as jnp
import jraph
import optax
from flax.training.train_state import TrainState

from gjp import bag_gae, edge_weight_decoder, edge_weight_gae, mpg_edge_weight


def test_edge_weight_gae(batch_graphs, mlp_kwargs):
    max_num_nodes = jnp.max(batch_graphs.n_node) + 1
    multi_edge_repeat = bag_gae.find_multi_edge_repeat(batch_graphs)

    mlp_stack = [1, 4, 32, 16, 4, 2]
    mpg_stack = [[2], [4, 8], [16], [2]]

    model = edge_weight_gae.EdgeWeightGAE(
        max_num_nodes=max_num_nodes,
        encoder_stack=mpg_stack,
        node_stack=mlp_stack,
        edge_stack=mlp_stack,
        decoder_mpg_stack=mpg_stack,
        multi_edge_repeat=multi_edge_repeat,
        mlp_kwargs=mlp_kwargs,
    )
    rng = jax.random.key(234)
    rng, rng_split_a, rng_split_b = jax.random.split(rng, 3)

    params = model.init({"params": rng, "reparametrize": rng_split_a, "dropout": rng_split_b}, batch_graphs)
    rng, rng_split_a, rng_split_b = jax.random.split(rng, 3)

    apply_model = jax.jit(lambda x, y, z, w: model.apply(x, y, rngs={"reparametrize": z, "dropout": w}))
    out_graphs, edge_weights, mu, sigma = apply_model(params, batch_graphs, rng_split_a, rng_split_b)
    rng, rng_split = jax.random.split(rng)

    out_graphs = edge_weight_decoder.make_graph_sparse(out_graphs, edge_weights)

    out_unbatch = jraph.unbatch(out_graphs)

    for i, in_graph in enumerate(jraph.unbatch(batch_graphs)):
        out_graph = out_unbatch[i]
        assert out_graph.n_node == in_graph.n_node


def test_ew_loss_function(batch_graphs, mlp_kwargs):
    max_num_nodes = jnp.max(batch_graphs.n_node) + 1
    multi_edge_repeat = bag_gae.find_multi_edge_repeat(batch_graphs)

    mlp_stack = [1, 4, 32, 16, 4, batch_graphs.nodes.shape[1]]
    mpg_stack = [[2], [4, 8], [16], [2]]

    model = edge_weight_gae.EdgeWeightGAE(
        max_num_nodes=max_num_nodes,
        encoder_stack=mpg_stack,
        node_stack=mlp_stack,
        edge_stack=mlp_stack,
        decoder_mpg_stack=mpg_stack,
        multi_edge_repeat=multi_edge_repeat,
        mlp_kwargs=mlp_kwargs,
    )

    rng = jax.random.key(72)
    rng, rng_split_a, rng_split_b = jax.random.split(rng, 3)

    params = model.init({"params": rng, "reparametrize": rng_split_a, "dropout": rng_split_b}, batch_graphs)
    rng, rng_split_a, rng_split_b = jax.random.split(rng, 3)
    tx = optax.adam(learning_rate=1e-3)
    opt_state = tx.init(params)
    train_state = TrainState(step=0, apply_fn=model.apply, params=params, tx=tx, opt_state=opt_state)
    train_state.apply_fn(train_state.params, batch_graphs, rngs={"reparametrize": rng_split_a, "dropout": rng_split_b})

    metric_model = mpg_edge_weight.MessagePassingEW(
        node_feature_sizes=mpg_stack,
        edge_feature_sizes=mpg_stack,
        global_feature_sizes=mpg_stack,
        mean_instead_of_sum=False,
    )
    rng, rng_split = jax.random.split(rng)
    metric_params = metric_model.init(rng_split, batch_graphs)
    metric_state = TrainState(apply_fn=metric_model.apply, params=metric_params, step=None, tx=None, opt_state=None)

    partial_step = partial(edge_weight_gae.train_step, metric_state=metric_state)
    func = jax.jit(partial_step)

    for _ in range(3):
        train_state, train_loss, train_recon, train_kl, train_sharp, train_n_edge, test_recon, test_kl, test_sharp, test_n_edge = func(batch_graphs, batch_graphs, train_state, rng=rng_split)
        rng, rng_split = jax.random.split(rng)

        print(train_loss, train_recon, train_kl, train_sharp, train_n_edge, test_recon, test_kl, test_sharp, test_n_edge)
