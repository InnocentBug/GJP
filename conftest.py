import os
import tempfile

import pytest


@pytest.fixture(scope="session", autouse=True)
def ensure_tempfile():
    original_path = os.getcwd()
    # jax.config.update("jax_platform_name", "cpu")
    with tempfile.TemporaryDirectory() as tmp_path:
        os.chdir(tmp_path)
        yield original_path, tmp_path
