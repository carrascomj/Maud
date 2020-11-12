"""Unit tests for sampling functions."""

import json
import os

from numpy.testing import assert_equal

from maud import io, sampling


here = os.path.dirname(__file__)
data_path = os.path.join(here, "../data")


def test_get_input_data():
    """Test that the function get_input_data behaves as expected."""
    toml_input_path = os.path.join(data_path, "linear.toml")
    mi = io.load_maud_input_from_toml(toml_input_path)
    expected = json.load(open(os.path.join(data_path, "linear.json"), "r"))
    actual = sampling.get_input_data(mi, 1e-06, 1e-06, int(1e9), 1, 500)
    assert actual.keys() == expected.keys()
    for k in actual.keys():
        print("*" * 8 + " " + k + " " + "*" * 8)
        print(actual[k])
        print(expected[k])
        print("\n")
        assert_equal(actual[k], expected[k], err_msg=f"{k} is different from expected.")
