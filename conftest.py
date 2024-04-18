import os
import tempfile

import jax
import pytest


@pytest.fixture(scope="session", autouse=True)
def ensure_tempfile():
    jax.config.update("jax_platform_name", "cpu")
    with tempfile.TemporaryDirectory() as tmpdirname:
        os.chdir(tmpdirname)
        yield None
