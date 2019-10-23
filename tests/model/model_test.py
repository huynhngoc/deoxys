from deoxys.model import Model
from deoxys.utils import read_file


def test_load():
    pass


def test_save():
    pass


def test_from_config():
    config = read_file('tests/data/sequential-config.json')
    Model.from_config(config, shape=(28, 28, 1))
