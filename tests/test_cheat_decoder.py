import hashlib

import jax
import jax.numpy as jnp
import jraph
import optax
import pytest
from flax.training.train_state import TrainState

from gjp import cheat_decoder, metric_util


@pytest.mark.parametrize("seed", [11, 42, 45])
def test_cheat_decoder(ensure_tempfile, seed):
    rng = jax.random.key(seed)
    rng, use_rng = jax.random.split(rng)

    model_args = {}

    max_num_nodes = int(jax.random.randint(use_rng, (1,), 3, 50)[0])
    model_args["max_nodes"] = max_num_nodes
    rng, use_rng = jax.random.split(rng)
    max_num_edges = int(jax.random.randint(use_rng, (1,), max_num_nodes, 2 * max_num_nodes)[0])
    model_args["max_edges"] = max_num_edges
    rng, use_rng = jax.random.split(rng)

    arch_stack_size = int(jax.random.randint(use_rng, (1,), 1, 5)[0])
    rng, use_rng = jax.random.split(rng)
    arch_stack = [int(i) for i in jax.random.randint(use_rng, (arch_stack_size,), 1, 64)]
    model_args["arch_stack"] = arch_stack
    rng, use_rng = jax.random.split(rng)

    node_stack_size = int(jax.random.randint(use_rng, (1,), 1, 5)[0])
    rng, use_rng = jax.random.split(rng)
    model_args["node_stack"] = [int(i) for i in jax.random.randint(use_rng, (node_stack_size,), 1, 64)]
    rng, use_rng = jax.random.split(rng)

    node_features = int(jax.random.randint(use_rng, (1,), 1, 10)[0])
    model_args["node_stack"] += [node_features]
    rng, use_rng = jax.random.split(rng)

    edge_stack_size = int(jax.random.randint(use_rng, (1,), 1, 10)[0])
    rng, use_rng = jax.random.split(rng)
    model_args["edge_stack"] = [int(i) for i in jax.random.randint(use_rng, (edge_stack_size,), 1, 64)]
    rng, use_rng = jax.random.split(rng)

    edge_features = int(jax.random.randint(use_rng, (1,), 2, 10)[0])
    model_args["edge_stack"] += [edge_features]
    rng, use_rng = jax.random.split(rng)

    input_size = int(jax.random.randint(use_rng, (1,), 2, 4)[0])
    rng, use_rng = jax.random.split(rng)
    input_dim = int(jax.random.randint(use_rng, (1,), 2, 10)[0])
    rng, use_rng = jax.random.split(rng)

    data_input = jax.random.normal(use_rng, (input_dim, input_size))
    rng, use_rng = jax.random.split(rng)

    n_node = jax.random.randint(use_rng, (input_size,), 0, max_num_nodes - 1)
    rng, use_rng = jax.random.split(rng)
    n_edge = jax.random.randint(use_rng, (input_size,), 0, max_num_edges - 1)
    rng, use_rng = jax.random.split(rng)

    test_input = jnp.vstack((data_input, n_node, n_edge)).transpose()

    print(model_args)

    original_path, _ = ensure_tempfile
    model = cheat_decoder.CheatDecoder(**model_args)

    params = model.init(use_rng, test_input)
    rng, use_rng = jax.random.split(rng)

    apply_model = jax.jit(lambda x, y: model.apply(x, y))
    apply_model(params, test_input)  # Tracing run
    new_graph = apply_model(params, test_input)

    assert new_graph.nodes.shape == (test_input.shape[0] * max_num_nodes, node_features)
    assert new_graph.edges.shape == (test_input.shape[0] * max_num_edges, edge_features)
    assert new_graph.n_node.shape == (2 * test_input.shape[0],)
    assert new_graph.n_edge.shape == (2 * test_input.shape[0],)
    assert new_graph.globals.shape == (2 * test_input.shape[0], test_input.shape[1])

    assert jnp.sum(new_graph.n_node) == test_input.shape[0] * max_num_nodes
    assert jnp.sum(new_graph.n_edge) == test_input.shape[0] * max_num_edges

    assert new_graph.senders.shape == (test_input.shape[0] * max_num_edges,)
    assert new_graph.receivers.shape == (test_input.shape[0] * max_num_edges,)

    unbatch_new_graph = jraph.unbatch(new_graph)
    assert len(unbatch_new_graph) == 2 * test_input.shape[0]

    m = hashlib.shake_256()
    m.update(test_input.tobytes())
    # metric_util.svg_graph_list(unbatch_new_graph, filename=os.path.join(original_path, f"test_cheat_decoder{m.hexdigest(3)}.pdf"))

    # Run some test to make sure the are legit
    for i in range(test_input.shape[0]):
        assert unbatch_new_graph[i * 2].n_node[0] == test_input[i, -2]
        assert unbatch_new_graph[i * 2].n_edge[0] <= test_input[i, -1]

        assert unbatch_new_graph[i * 2 + 1].n_node[0] == max_num_nodes - test_input[i, -2]
        assert unbatch_new_graph[i * 2 + 1].n_edge[0] >= max_num_edges - test_input[i, -1]
        nx_graphA = metric_util.nx_from_jraph(unbatch_new_graph[i * 2])
        assert len(nx_graphA.nodes()) == unbatch_new_graph[i * 2].n_node[0]
        assert len(nx_graphA.edges()) == unbatch_new_graph[i * 2].n_edge[0]

        nx_graphB = metric_util.nx_from_jraph(unbatch_new_graph[i * 2 + 1])
        assert len(nx_graphB.nodes()) == unbatch_new_graph[i * 2 + 1].n_node[0]
        assert len(nx_graphB.edges()) == unbatch_new_graph[i * 2 + 1].n_edge[0]

        assert unbatch_new_graph[i * 2].senders.size == unbatch_new_graph[i * 2].n_edge[0]
        assert unbatch_new_graph[i * 2 + 1].senders.size == unbatch_new_graph[i * 2 + 1].n_edge[0]

        assert unbatch_new_graph[i * 2].receivers.size == unbatch_new_graph[i * 2].n_edge[0]
        assert unbatch_new_graph[i * 2 + 1].receivers.size == unbatch_new_graph[i * 2 + 1].n_edge[0]

        assert unbatch_new_graph[i * 2].n_edge[0] >= 0
        assert unbatch_new_graph[i * 2 + 1].n_edge[0] >= 0

        if unbatch_new_graph[i * 2].n_edge[0] > 0:
            assert jnp.max(unbatch_new_graph[i * 2].senders) < unbatch_new_graph[i * 2].n_node[0]
            assert jnp.max(unbatch_new_graph[i * 2].receivers) < unbatch_new_graph[i * 2].n_node[0]

            assert jnp.min(unbatch_new_graph[i * 2].senders) >= 0
            assert jnp.min(unbatch_new_graph[i * 2].receivers) >= 0

        if unbatch_new_graph[i * 2 + 1].n_edge[0] > 0:
            assert jnp.allclose(unbatch_new_graph[i * 2 + 1].senders, jnp.zeros(unbatch_new_graph[i * 2 + 1].n_edge[0]))
            assert jnp.allclose(unbatch_new_graph[i * 2 + 1].receivers, jnp.zeros(unbatch_new_graph[i * 2 + 1].n_edge[0]))


@pytest.mark.parametrize("seed", [17, 448, 126])
def test_grad_cheat_decoder(seed):
    rng = jax.random.key(seed)
    rng, use_rng = jax.random.split(rng)

    model_args = {}

    max_num_nodes = int(jax.random.randint(use_rng, (1,), 3, 50)[0])
    model_args["max_nodes"] = max_num_nodes
    rng, use_rng = jax.random.split(rng)
    max_num_edges = int(jax.random.randint(use_rng, (1,), max_num_nodes, 2 * max_num_nodes)[0])
    model_args["max_edges"] = max_num_edges
    rng, use_rng = jax.random.split(rng)

    arch_stack_size = int(jax.random.randint(use_rng, (1,), 5, 10)[0])
    rng, use_rng = jax.random.split(rng)
    arch_stack = [int(i) for i in jax.random.randint(use_rng, (arch_stack_size,), 32, 64)]
    model_args["arch_stack"] = arch_stack
    rng, use_rng = jax.random.split(rng)

    node_stack_size = int(jax.random.randint(use_rng, (1,), 3, 5)[0])
    rng, use_rng = jax.random.split(rng)
    model_args["node_stack"] = [int(i) for i in jax.random.randint(use_rng, (node_stack_size,), 32, 64)]
    rng, use_rng = jax.random.split(rng)

    node_features = int(jax.random.randint(use_rng, (1,), 1, 10)[0])
    model_args["node_stack"] += [node_features]
    rng, use_rng = jax.random.split(rng)

    edge_stack_size = int(jax.random.randint(use_rng, (1,), 3, 10)[0])
    rng, use_rng = jax.random.split(rng)
    model_args["edge_stack"] = [int(i) for i in jax.random.randint(use_rng, (edge_stack_size,), 32, 64)]
    rng, use_rng = jax.random.split(rng)

    edge_features = int(jax.random.randint(use_rng, (1,), 2, 10)[0])
    model_args["edge_stack"] += [edge_features]
    rng, use_rng = jax.random.split(rng)

    input_size = int(jax.random.randint(use_rng, (1,), 2, 4)[0])
    rng, use_rng = jax.random.split(rng)
    input_dim = int(jax.random.randint(use_rng, (1,), 2, 10)[0])
    rng, use_rng = jax.random.split(rng)

    data_input = jax.random.normal(use_rng, (input_dim, input_size))
    rng, use_rng = jax.random.split(rng)

    n_node = jax.random.randint(use_rng, (input_size,), 0, max_num_nodes - 1)
    rng, use_rng = jax.random.split(rng)
    n_edge = jax.random.randint(use_rng, (input_size,), 0, max_num_edges - 1)
    rng, use_rng = jax.random.split(rng)

    test_input = jnp.vstack((data_input, n_node, n_edge)).transpose()

    print(model_args)

    model = cheat_decoder.CheatDecoder(**model_args)

    params = model.init(use_rng, test_input)
    rng, use_rng = jax.random.split(rng)

    tx = optax.adam(learning_rate=1e-3)
    opt_state = tx.init(params)
    state = TrainState(params=params, step=0, tx=tx, opt_state=opt_state, apply_fn=model.apply)

    a_graphs = model.apply(state.params, test_input)
    a_graphs = cheat_decoder.indexify_graph(a_graphs)
    b_graphs = jraph.unbatch(a_graphs)
    c_graphs = jraph.batch(b_graphs[::2])
    pad_graphs = cheat_decoder.batch_graph_arrays(c_graphs, model.max_edges, model.max_nodes)

    def loss_fn(params, state, x):
        new_graphs = state.apply_fn(params, x)

        diff_graph = cheat_decoder.make_abs_diff_graph(new_graphs, pad_graphs)

        loss = jnp.mean(diff_graph.nodes) + jnp.max(diff_graph.nodes)
        loss += jnp.mean(diff_graph.edges) + jnp.max(diff_graph.edges)
        loss += jnp.mean(diff_graph.senders) + jnp.max(diff_graph.senders)
        loss += jnp.mean(diff_graph.receivers) + jnp.max(diff_graph.receivers)

        return loss

    val, grad = jax.value_and_grad(loss_fn)(state.params, state, test_input)
    # assert val > 0.2

    # Ensure we have gradients on all our params
    for leaf in jax.tree_util.tree_leaves(grad):
        assert jnp.sum(jnp.abs(leaf)) > 0

    def train_step(state, x):
        a_graphs = model.apply(state.params, x)

        val, grad = jax.value_and_grad(loss_fn)(state.params, state, x)
        state = state.apply_gradients(grads=grad)
        return state, val

    jit_step = jax.jit(train_step)

    for i in range(10000):
        state, val = jit_step(state, test_input)
        if i % 1000 == 0:
            print(i, val)

    a_graphs = model.apply(state.params, test_input)
    idx_graphs = cheat_decoder.indexify_graph(a_graphs)
    # print(jnp.max(jnp.abs(a_graphs.senders - idx_graphs.senders)), a_graphs.senders - idx_graphs.senders)
    # print(jnp.max(jnp.abs(a_graphs.receivers - idx_graphs.receivers)), a_graphs.receivers - idx_graphs.receivers)
    assert jnp.max(jnp.abs(a_graphs.senders - idx_graphs.senders)) < 0.2
    assert jnp.max(jnp.abs(a_graphs.receivers - idx_graphs.receivers)) < 0.2
