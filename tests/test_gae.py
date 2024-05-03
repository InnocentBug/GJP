import jax

from gjp import GraphData, batch_list, convert_to_jraph, gae, metric_util


def test_gae():
    with GraphData(".test_batching") as dataset:
        train, _ = dataset.get_test_train(15, 0, 5, 11)
        train_jraph = convert_to_jraph(train)
        for i in range(15):
            similar_data = dataset.get_similar_feature_graphs(train_jraph[i], 2)
            train_jraph += similar_data

        num_nodes_train, num_edges_train = metric_util._count_nodes_edges(train_jraph)

        node_batch_size = num_nodes_train + 1
        edge_batch_size = num_edges_train + 1
        batch_train = batch_list(train_jraph, node_batch_size, edge_batch_size)[0]

        model = gae.GAE(
            encoder_stack=[[2], [4, 8], [16], [2]],
            max_num_nodes=20,
            max_num_edges=40,
            init_stack=[8, 4, 16, 32],
            init_features=4,
            prob_stack=[[4], [8], [16]],
            feature_stack=[[4], [8], [16]],
            node_features=batch_train.nodes.shape[1],
            edge_features=batch_train.edges.shape[1],
            max_num_graph=len(train_jraph),
        )
        rng = jax.random.key(234)
        rng, rng_split = jax.random.split(rng)

        params = model.init({"params": rng, "reparametrize": rng_split}, batch_train)
