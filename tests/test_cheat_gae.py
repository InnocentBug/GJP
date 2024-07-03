from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph
import optax
from flax.training.train_state import TrainState

from gjp import cheat_decoder, cheat_gae

MLP_KWARGS = {"dropout_rate": 0.1, "deterministic": False, "activation": nn.sigmoid}


def test_cheat_gae(batch_graphs):
    max_num_nodes = jnp.max(batch_graphs.n_node[:-1]) + 1
    max_num_edges = jnp.max(batch_graphs.n_edge[:-1]) + 1

    mlp_stack = [1, 4, 7, 2, int(batch_graphs.nodes.shape[1])]
    mpg_stack = [[2], [4, 8], [16]]

    model = cheat_gae.CheatGAE(
        max_nodes=max_num_nodes,
        max_edges=max_num_edges,
        arch_stack=mlp_stack,
        node_stack=mlp_stack,
        edge_stack=mlp_stack,
        encoder_stack=mpg_stack,
        mlp_kwargs=MLP_KWARGS,
    )
    rng = jax.random.key(234)
    rng, rng_split_a, rng_split_b = jax.random.split(rng, 3)

    params = model.init({"params": rng, "reparametrize": rng_split_a, "dropout": rng_split_b}, batch_graphs)
    rng, rng_split_a, rng_split_b = jax.random.split(rng, 3)

    apply_model = jax.jit(lambda x, y, z, w: model.apply(x, y, rngs={"reparametrize": z, "dropout": w}))
    out_graphs, mu, log_sigma = apply_model(params, batch_graphs, rng_split_a, rng_split_b)
    rng, rng_split = jax.random.split(rng)

    out_unbatch = jraph.unbatch(out_graphs)

    for i, in_graph in enumerate(jraph.unbatch(batch_graphs)[:-1]):
        out_graph = cheat_decoder.indexify_graph(out_unbatch[2 * i])
        assert out_graph.n_node == in_graph.n_node
        assert out_graph.n_edge[0] == in_graph.n_edge[0]
        assert out_graph.globals.shape[1] == mpg_stack[-1][-1] + 2
        assert out_graph.globals[0, -1] == in_graph.n_edge[0]
        assert out_graph.globals[0, -2] == in_graph.n_node[0]


def test_cheat_loss_function(batch_graphs):
    max_num_nodes = jnp.max(batch_graphs.n_node[:-1]) + 1
    max_num_edges = jnp.max(batch_graphs.n_edge[:-1]) + 1

    mlp_stack = [1, 4, 7, 2, int(batch_graphs.nodes.shape[1])]
    mpg_stack = [[2], [4, 8], [16]]

    model = cheat_gae.CheatGAE(
        max_nodes=max_num_nodes,
        max_edges=max_num_edges,
        arch_stack=mlp_stack,
        node_stack=mlp_stack,
        edge_stack=mlp_stack,
        encoder_stack=mpg_stack,
        mlp_kwargs=MLP_KWARGS,
    )
    rng = jax.random.key(234)
    rng, rng_split_a, rng_split_b = jax.random.split(rng, 3)

    params = model.init({"params": rng, "reparametrize": rng_split_a, "dropout": rng_split_b}, batch_graphs)

    rng, rng_split_a, rng_split_b = jax.random.split(rng, 3)
    tx = optax.adam(learning_rate=1e-3)
    opt_state = tx.init(params)
    state = TrainState(step=0, apply_fn=model.apply, params=params, tx=tx, opt_state=opt_state)

    ref_train = cheat_decoder.batch_graph_arrays(batch_graphs, model.max_edges, model.max_nodes)
    partial_step = partial(cheat_gae.train_step, model=model, test_graph=batch_graphs, ref_test_graph=ref_train)
    func = jax.jit(partial_step)

    for i in range(3):
        rng, rng_split = jax.random.split(rng)
        state, train_loss, test_loss = func(state, train_graph=batch_graphs, ref_train_graph=ref_train, rng=rng_split)

        print(i, train_loss, test_loss)
