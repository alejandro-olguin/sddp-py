import json
import pathlib

import pytest

ORACLE_DIR = pathlib.Path(__file__).resolve().parent.parent / "reference" / "oracle"


def load_oracle(name: str) -> dict:
    with open(ORACLE_DIR / f"{name}.json") as io:
        return json.load(io)


@pytest.fixture
def oracle():
    return load_oracle
