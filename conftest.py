import os
import tempfile

import flax.linen as nn
import jax
import pytest

from gjp import GraphData, batch_list, convert_to_jraph, metric_util


@pytest.fixture(scope="session", autouse=True)
def ensure_tempfile():
    original_path = os.getcwd()
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_debug_nans", True)
    jax.config.update("jax_experimental_unsafe_xla_runtime_errors", True)
    with tempfile.TemporaryDirectory() as tmp_path:
        os.chdir(tmp_path)
        yield original_path, tmp_path


@pytest.fixture(scope="session")
def dataset(ensure_tempfile):
    with GraphData(".test_dataset", seed=34) as ds:
        yield ds


@pytest.fixture(scope="session")
def nx_graphs(dataset):
    train, test = dataset.get_test_train(10, 5, 5, 11)
    train += test
    return train


@pytest.fixture(scope="session")
def jraph_graphs(nx_graphs):
    return convert_to_jraph(nx_graphs)


@pytest.fixture(scope="session")
def similar_graphs(jraph_graphs, dataset):
    return jraph_graphs + dataset.get_similar_feature_graphs(jraph_graphs[0], 5)


@pytest.fixture(scope="session")
def batch_graphs(similar_graphs):
    num_nodes, num_edges = metric_util._count_nodes_edges(similar_graphs)
    return batch_list(similar_graphs, num_nodes + 1, num_edges + 1)[0]


@pytest.fixture(scope="session")
def jax_rng():
    return jax.random.key(42)


@pytest.fixture
def mlp_kwargs():
    mlp_kwargs = {"dropout_rate": 0.1, "deterministic": False, "activation": nn.sigmoid}
    return mlp_kwargs
