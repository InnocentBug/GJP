import jax

from gjp import GraphData, MessagePassing, convert_to_jraph


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

        out_graph = model.apply(params, test_jraph[0])
        assert out_graph.nodes.shape[1] == final_node_size
        assert out_graph.edges.shape[1] == final_edge_size
        assert out_graph.globals.shape[1] == final_global_size
