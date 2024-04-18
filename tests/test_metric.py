import os

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from gjp import GraphData, MessagePassing, batch_list, convert_to_jraph, metric_util

jax.config.update("jax_platform_name", "cpu")


def test_small_metric_model():
    params = metric_util.run_parameter(
        shelf_path=".test_small_metric_model",
        mlp_stack=[[1], [4], [2]],
        stepA=2,
        stepB=2,
        min_nodes=8,
        max_nodes=10,
        train_size=10,
        test_size=3,
        extra_feature=2,
        num_batch_shuffle=2,
        seed=None,
        node_pad=100,
        edge_pad=200,
        checkpoint_path="./",
        checkpoint_every=1,
        norm=False,
    )

    params = metric_util.run_parameter(
        shelf_path=".test_small_metric_model",
        mlp_stack=[[1], [4], [2]],
        stepA=3,
        stepB=4,
        min_nodes=8,
        max_nodes=10,
        train_size=10,
        test_size=3,
        extra_feature=2,
        num_batch_shuffle=2,
        seed=None,
        node_pad=100,
        edge_pad=200,
        checkpoint_path="./",
        checkpoint_every=1,
        norm=True,
        from_checkpoint="./1",
        epoch_offset=3,
    )

    # Validating run
    checkpointer = ocp.PyTreeCheckpointer()
    params = checkpointer.restore(os.path.abspath("./1"))
    node_pad = 400
    edge_pad = 600
    mlp_stack = [[1], [4], [2]]
    model = MessagePassing(mlp_stack, mlp_stack, mlp_stack, num_nodes=node_pad)

    with GraphData(".test_small_metric_model") as dataset:
        train, test = dataset.get_test_train(15, 10, 7, 11)
        data = convert_to_jraph(train + test)
        similar_data = dataset.get_similar_feature_graphs(data[0], 5)
        data = batch_list(data + similar_data, node_pad, edge_pad)

        assert len(data) == 1
        data = data[0]

        idx = metric_util.loss_function_where(params, data, model, 1e-10)
        out_graph = model.apply(params, data)
        print(out_graph.globals)
        print(jnp.sqrt(jnp.sum(out_graph.globals**2, axis=-1)))
        for i, j in zip(idx[0], idx[1]):
            print(i, j, out_graph.globals[i], out_graph.globals[j], out_graph.n_node[i], out_graph.n_node[j])
        assert len(idx[0]) == 0
