import hashlib

import jax
import jax.numpy as jnp
import jraph
import pytest

from gjp import decoder, metric_util

jax.config.update("jax_platform_name", "cpu")


@pytest.mark.parametrize("max_num_nodes, max_num_edges, n_edge_features", [(3, 6, 3), (7, 3, 5), (50, 150, 7)])
def test_initial_edge_decoder(max_num_nodes, max_num_edges, n_edge_features):
    model = decoder.InitialEdgeDecoder(mlp_stack=[16, 32, 64], max_num_nodes=max_num_nodes, max_num_edges=max_num_edges, n_edge_features=n_edge_features, max_edge_node=max_num_edges * max_num_nodes)

    test_input = jnp.asarray([[1, 5, 8, 6.1, 44], [1, 5, 8, 6.1, 4.4]])

    rng = jax.random.key(15)
    rng, init_rng = jax.random.split(rng)
    params = model.init(init_rng, test_input)

    apply_model = jax.jit(lambda x: model.apply(params, x))

    senders, receivers, features = apply_model(test_input)

    assert senders.shape == (test_input.shape[0], max_num_edges)
    assert receivers.shape == (test_input.shape[0], max_num_edges)

    for i in range(test_input.shape[0]):
        assert jnp.min(senders[i]) >= i * max_num_nodes
        assert jnp.max(senders[i]) <= (i + 1) * max_num_nodes
        assert jnp.min(receivers[i]) >= i * max_num_nodes
        assert jnp.max(receivers[i]) <= (i + 1) * max_num_nodes

    assert features.shape == (2, max_num_edges, n_edge_features)


@pytest.mark.parametrize("max_num_nodes, n_node_features", [(3, 1), (6, 4), (150, 16)])
def test_initial_node_decoder(max_num_nodes, n_node_features):
    model = decoder.InitialNodeDecoder(mlp_stack=[15, 76, 651], max_num_nodes=max_num_nodes, n_node_features=n_node_features)

    test_input = jnp.asarray([[1, 5, 8, 6.1, 44], [1, 5, 8, 6.1, 44]])
    rng = jax.random.key(15)
    rng, init_rng = jax.random.split(rng)
    params = model.init(init_rng, test_input)

    apply_model = jax.jit(lambda x: model.apply(params, x))
    initial_node_features = apply_model(test_input)
    assert initial_node_features.shape == (test_input.shape[0], max_num_nodes, n_node_features)


@pytest.mark.parametrize("init_edge_features, init_node_features, max_num_nodes, max_num_edges", [(3, 2, 4, 8), (7, 9, 50, 100)])
def test_initial_graph_decoder(init_edge_features, init_node_features, max_num_nodes, max_num_edges):

    model = decoder.InitialGraphDecoder(
        init_edge_stack=[2, 4, 8], init_edge_features=init_edge_features, init_node_stack=[5, 10, 7], init_node_features=init_node_features, max_num_nodes=max_num_nodes, max_num_edges=max_num_edges, max_edge_node=max_num_edges * max_num_nodes
    )

    test_input = jnp.asarray([[0.1, 0.2, 0.3, 5, 6], [0.5, 0.2, 0.4, 7, 14]])

    rng = jax.random.key(15)
    rng, init_rng = jax.random.split(rng)
    params = model.init(init_rng, test_input)

    apply_model = jax.jit(lambda x: model.apply(params, x))
    out_graph = apply_model(test_input)
    out_batch = jraph.unbatch(out_graph)
    assert len(out_batch) == test_input.shape[0]
    for i, graph in enumerate(out_batch):
        assert graph.nodes.shape == (max_num_nodes, init_node_features)
        assert graph.edges.shape == (max_num_edges, init_edge_features)

        assert graph.senders.shape == (max_num_edges,)
        assert jnp.min(graph.senders) >= 0
        assert jnp.max(graph.senders) <= max_num_nodes

        assert graph.receivers.shape == (max_num_edges,)
        assert jnp.min(graph.receivers) >= 0
        assert jnp.max(graph.receivers) <= max_num_nodes

        assert jnp.allclose(graph.globals, test_input[i])


@pytest.mark.parametrize("seed", [11, 42, 45])
def test_graph_decoder(ensure_tempfile, seed):
    rng = jax.random.key(seed)
    rng, use_rng = jax.random.split(rng)

    model_args = {}

    max_num_nodes = int(jax.random.randint(use_rng, (1,), 3, 50)[0])
    model_args["max_num_nodes"] = max_num_nodes
    rng, use_rng = jax.random.split(rng)
    max_num_edges = int(jax.random.randint(use_rng, (1,), max_num_nodes, 2 * max_num_nodes)[0])
    model_args["max_num_edges"] = max_num_edges
    rng, use_rng = jax.random.split(rng)

    init_edge_stack_size = int(jax.random.randint(use_rng, (1,), 1, 5)[0])
    rng, use_rng = jax.random.split(rng)
    init_edge_stack = [int(i) for i in jax.random.randint(use_rng, (init_edge_stack_size,), 1, 64)]
    model_args["init_edge_stack"] = init_edge_stack
    rng, use_rng = jax.random.split(rng)

    init_edge_features = int(jax.random.randint(use_rng, (1,), 2, 10)[0])
    rng, use_rng = jax.random.split(rng)
    model_args["init_edge_features"] = init_edge_features

    node_stack_size = int(jax.random.randint(use_rng, (1,), 1, 5)[0])
    rng, use_rng = jax.random.split(rng)
    model_args["init_node_stack"] = [int(i) for i in jax.random.randint(use_rng, (node_stack_size,), 1, 64)]
    rng, use_rng = jax.random.split(rng)

    model_args["init_node_features"] = int(jax.random.randint(use_rng, (1,), 1, 10)[0])
    rng, use_rng = jax.random.split(rng)

    prob_stack_length = int(jax.random.randint(use_rng, (1,), 1, 10)[0])
    rng, use_rng = jax.random.split(rng)

    prob_node_depth = int(jax.random.randint(use_rng, (1,), 1, 4)[0])
    rng, use_rng = jax.random.split(rng)
    prob_edge_depth = int(jax.random.randint(use_rng, (1,), 1, 4)[0])
    rng, use_rng = jax.random.split(rng)

    model_args["prob_node_stack"] = jax.random.randint(use_rng, (prob_stack_length, prob_node_depth), 1, 32).tolist()
    rng, use_rng = jax.random.split(rng)
    model_args["prob_edge_stack"] = jax.random.randint(use_rng, (prob_stack_length, prob_edge_depth), 1, 32).tolist()
    rng, use_rng = jax.random.split(rng)

    feature_stack_length = int(jax.random.randint(use_rng, (1,), 1, 10)[0])
    rng, use_rng = jax.random.split(rng)

    feature_node_depth = int(jax.random.randint(use_rng, (1,), 1, 4)[0])
    rng, use_rng = jax.random.split(rng)
    feature_edge_depth = int(jax.random.randint(use_rng, (1,), 1, 4)[0])
    rng, use_rng = jax.random.split(rng)

    model_args["feature_node_stack"] = jax.random.randint(use_rng, (feature_stack_length, feature_node_depth), 1, 32).tolist()
    rng, use_rng = jax.random.split(rng)
    model_args["feature_edge_stack"] = jax.random.randint(use_rng, (feature_stack_length, feature_edge_depth), 1, 32).tolist()
    rng, use_rng = jax.random.split(rng)

    model_args["max_edge_node"] = model_args["max_num_nodes"] * model_args["max_num_edges"]

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

    model_args["max_num_graphs"] = test_input.shape[0] + 1

    print(model_args)

    original_path, _ = ensure_tempfile
    model = decoder.GraphDecoder(**model_args)

    params = model.init(use_rng, test_input)
    rng, use_rng = jax.random.split(rng)

    apply_model = jax.jit(lambda x, y: model.apply(x, y))
    new_graph = apply_model(params, test_input)
    unbatch_new_graph = jraph.unbatch(new_graph)
    assert len(unbatch_new_graph) == 2 * test_input.shape[0]

    m = hashlib.shake_256()
    m.update(test_input.tobytes())
    # metric_util.svg_graph_list(unbatch_new_graph, filename=os.path.join(original_path, f"test_graph_decoder{m.hexdigest(3)}.pdf"))

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
