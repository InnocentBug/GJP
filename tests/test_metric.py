import jax
import numpy as np
import optax

from gjp import (
    GraphData,
    MessagePassing,
    batch_list,
    change_global_jraph_to_props,
    convert_to_jraph,
    metric_util,
)


def test_small_metric_model():
    # Consider seed for less brittleness
    seed = None

    with GraphData("./.test_small_metric_model", seed=seed) as dataset:
        train, test = dataset.get_test_train(25, 5, 8, 10)
        node_batch_size = 100
        edge_batch_size = 200

        train_jraph = convert_to_jraph(train)
        test_jraph = convert_to_jraph(test)

        # Add graphs with the same architecture but different initial features
        similar_train_graphs = dataset.get_similar_feature_graphs(train_jraph[0], 5)
        similar_test_graphs = dataset.get_similar_feature_graphs(test_jraph[0], 2)

        train_jraph += similar_train_graphs
        test_jraph += similar_test_graphs

        # Batch the test into a single graph
        batch_test = change_global_jraph_to_props(batch_list(test_jraph, node_batch_size, edge_batch_size), node_batch_size)
        assert len(batch_test) == 1
        batch_test = batch_test[0]

        # Use different combinations of batching the training data
        np_rng = np.random.default_rng(seed)
        num_batch_shuffle = 2
        batch_shuffles = []
        for _ in range(num_batch_shuffle):
            batched_train_data = change_global_jraph_to_props(batch_list(train_jraph, node_batch_size, edge_batch_size), node_batch_size)
            assert len(batched_train_data) > 1
            batch_shuffles.append(batched_train_data)
            np_rng.shuffle(train_jraph)

        # Build the Message Passing Model, with 3 layers, and each layer uses a single MLP layer only
        node_stack = [[2], [3], [4]]
        # Same dimension (but different weights) for all MLP in the model
        edge_stack = node_stack
        global_stack = node_stack

        model = MessagePassing(edge_stack, node_stack, global_stack, num_nodes=node_batch_size)
        # Random parameter init
        rng = jax.random.key(np_rng.integers(50000))
        rng, init_rng = jax.random.split(rng)
        params = model.init(init_rng, batch_test)

        tx = optax.adam(learning_rate=1e-2)
        opt_state = tx.init(params)

        # Learning loop
        for i in range(3):
            params, tx, opt_state = metric_util.train_model(batch_shuffles[i % num_batch_shuffle], batch_test, 2, model, params, tx, opt_state)

        # Check that all graphs have different embedding in test set.
        idx = metric_util.loss_function_where(params, batch_test, model, 1e-10)

        assert len(idx[0]) == 0
        assert len(idx[1]) == 0
