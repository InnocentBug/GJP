import jraph
import networkx as nx
import numpy as np

from gjp import GraphData, change_global_jraph_to_props, convert_to_jraph, get_pad_graph


def test_init_dataset():
    with GraphData(".test_init_dataset") as dataset:
        dataset.ensure_num_random_graphs(4, 1)

    with GraphData(".test_init_dataset") as dataset:
        dataset.ensure_num_random_graphs(4, 1)

    with GraphData(".test_init_dataset", overwrite=True) as dataset:
        dataset.ensure_num_random_graphs(4, 1)

    with GraphData(".test_init_dataset", overwrite=True, seed=42) as dataset:
        dataset.ensure_num_random_graphs(4, 1)

    with GraphData(".test_init_dataset", overwrite=True, seed=42, min_edges=17, max_edges=32) as dataset:
        dataset.ensure_num_random_graphs(4, 1)


def test_random_data():
    with GraphData(".test_random_data") as dataset:
        dataset.ensure_num_random_graphs(4, 10)
        dataset.get_graph("random", 4, 7)


def test_similar_data():
    with GraphData(".test_similar_data") as dataset:
        dataset.ensure_num_similar_graphs(4, 10)
        dataset.get_graph("similar", 4, 8)


def test_training_set():
    with GraphData(".test_training_set") as dataset:
        train, test = dataset.get_test_train(10, 5, 5, 11)
        assert len(train) == 10
        assert len(test) == 5


def test_jraph_graphs():
    with GraphData(".test_jraph_graphs") as dataset:
        train, test = dataset.get_test_train(10, 5, 5, 11)

        max_node_pad = 32
        max_edges_pad = 64

        convert_to_jraph(train, max_node_pad, max_edges_pad)
        convert_to_jraph(test, max_node_pad, max_edges_pad)


def test_similar_feature():
    with GraphData(".test_jraph_graphs") as dataset:
        train, test = dataset.get_test_train(10, 5, 5, 11)
        max_node_pad = 32
        max_edges_pad = 64

        train_jraph = convert_to_jraph(train, max_node_pad, max_edges_pad)
        test_jraph = convert_to_jraph(test, max_node_pad, max_edges_pad)

        dataset.get_similar_feature_graphs(train_jraph[0], 20)
        train += [train[0]] * 20
        dataset.get_similar_feature_graphs(test_jraph[0], 10)
        test += [test[0]] * 10


def test_global_features():
    graph = nx.MultiDiGraph()
    for i in range(5):
        graph.add_node(i)

    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(0, 3)
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(3, 1)
    graph.add_edge(3, 0)
    graph.add_edge(3, 2)
    graph.add_edge(3, 0)
    graph.add_edge(1, 3)

    jraph_graph = convert_to_jraph([graph])[0]
    pad_graph = get_pad_graph(jraph_graph, 9, 11)

    batch_graph = jraph.batch([jraph_graph, pad_graph])

    new_graph = change_global_jraph_to_props([batch_graph], 7)[0]

    result = np.asarray(new_graph.globals)

    expected_result = np.array(
        [
            [
                3.3,
                3.3,
                3.3,
                2.6666667,
                5.0,
                10.0,
            ],
            [
                1.0,
                1.0,
                1.0,
                1.0,
                4.0,
                1.0,
            ],
        ]
    )

    assert np.allclose(result, expected_result)
