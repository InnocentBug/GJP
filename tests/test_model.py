import jax
import jax.numpy as jnp
import pytest

from gjp import GraphData, MessagePassing, batch_list, convert_to_jraph, metric_util


def test_simple_pass():
    with GraphData(".test_simple_pass") as dataset:
        train, test = dataset.get_test_train(10, 5, 5, 11)
        max_node_pad = 32
        max_edges_pad = 64

        train_jraph = convert_to_jraph(train, max_node_pad, max_edges_pad)
        test_jraph = convert_to_jraph(test, max_node_pad, max_edges_pad)

        dataset.get_similar_feature_graphs(train_jraph[0], 20)
        train += [train[0]] * 20
        dataset.get_similar_feature_graphs(test_jraph[0], 10)
        test += [test[0]] * 10

        final_node_size = 21
        final_edge_size = 34
        final_global_size = 42

        node_stack = [[4], [2], [final_node_size]]
        edge_stack = [[4], [2], [final_edge_size]]
        global_stack = [[4], [2], [final_global_size]]

        rng = jax.random.key(42)
        rng, init_rng = jax.random.split(rng)
        model = MessagePassing(node_stack, edge_stack, global_stack, num_nodes=max_node_pad)
        params = model.init(init_rng, train_jraph[0])

        apply_model = jax.jit(lambda x: model.apply(params, x))
        out_graph = apply_model(test_jraph[0])
        assert out_graph.nodes.shape[1] == final_node_size
        assert out_graph.edges.shape[1] == final_edge_size
        assert out_graph.globals.shape[1] == final_global_size


@pytest.mark.parametrize("mean_instead_of_sum", (True, False))
def test_mean_instead_of_sum(mean_instead_of_sum):
    with GraphData(".test_simple") as dataset:
        train, test = dataset.get_test_train(10, 5, 5, 11)

        train_jraph = convert_to_jraph(train)
        test_jraph = convert_to_jraph(test)

        dataset.get_similar_feature_graphs(train_jraph[0], 20)
        train += [train[0]] * 20
        dataset.get_similar_feature_graphs(test_jraph[0], 10)
        test += [test[0]] * 10

        num_nodes_train, num_edges_train = metric_util._count_nodes_edges(train_jraph)
        num_nodes_test, num_edges_test = metric_util._count_nodes_edges(test_jraph)
        node_batch_size = max(num_nodes_train, num_nodes_test) + 1
        edge_batch_size = max(num_edges_train, num_edges_test) + 1

        batch_test = batch_list(test_jraph, node_batch_size, edge_batch_size)[0]
        batch_train = batch_list(train_jraph, node_batch_size, edge_batch_size)[0]

        final_node_size = 21
        final_edge_size = 34
        final_global_size = 42

        node_stack = [[4], [2], [final_node_size]]
        edge_stack = [[4], [2], [final_edge_size]]
        global_stack = [[4], [2], [final_global_size]]

        rng = jax.random.key(42)
        rng, init_rng = jax.random.split(rng)
        model = MessagePassing(node_stack, edge_stack, global_stack, num_nodes=node_batch_size, mean_instead_of_sum=mean_instead_of_sum)
        params = model.init(init_rng, batch_test)

        apply_model = jax.jit(lambda x: model.apply(params, x))
        out_graph = apply_model(batch_train)
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
def test_stacks(edge_stack, node_stack, global_stack):
    with GraphData(".test_simple") as dataset:
        train, test = dataset.get_test_train(10, 5, 5, 11)

        train_jraph = convert_to_jraph(train)
        test_jraph = convert_to_jraph(test)

        dataset.get_similar_feature_graphs(train_jraph[0], 20)
        train += [train[0]] * 20
        dataset.get_similar_feature_graphs(test_jraph[0], 10)
        test += [test[0]] * 10

        num_nodes_train, num_edges_train = metric_util._count_nodes_edges(train_jraph)
        num_nodes_test, num_edges_test = metric_util._count_nodes_edges(test_jraph)
        node_batch_size = max(num_nodes_train, num_nodes_test) + 1
        edge_batch_size = max(num_edges_train, num_edges_test) + 1

        batch_test = batch_list(test_jraph, node_batch_size, edge_batch_size)[0]
        batch_train = batch_list(train_jraph, node_batch_size, edge_batch_size)[0]

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
        model = MessagePassing(node_stack, edge_stack, global_stack, num_nodes=node_batch_size)
        params = model.init(init_rng, batch_test)

        apply_model = jax.jit(lambda x: model.apply(params, x))
        out_graph = apply_model(batch_train)
        if node_stack:
            assert out_graph.nodes.shape[1] == final_node_size
        else:
            assert jnp.allclose(out_graph.nodes, batch_train.nodes)
        if edge_stack:
            assert out_graph.edges.shape[1] == final_edge_size
        else:
            assert jnp.allclose(out_graph.edges, batch_train.edges)
        if global_stack:
            assert out_graph.globals.shape[1] == final_global_size
        else:
            jnp.allclose(out_graph.globals, batch_train.globals)
