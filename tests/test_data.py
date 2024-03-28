from gjp import GraphData, convert_to_jraph


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
