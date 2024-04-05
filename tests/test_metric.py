import jax

from gjp import metric_util

jax.config.update("jax_platform_name", "cpu")


def test_small_metric_model():
    same_test_nodes = metric_util.run_parameter(
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
    )
    assert len(same_test_nodes) == 0
