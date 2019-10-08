"""Unit tests for io functions."""

import os

import maud.io as io


here = os.path.dirname(__file__)
data_path = os.path.join(here, "../data")


def test_load_maud_input_from_toml():
    """Test that the function load_maud_input_from_toml behaves as expected."""
    mi = io.load_maud_input_from_toml(os.path.join(data_path, "linear.toml"))
    assert mi.kinetic_model.reactions["r1"].stoichiometry == {"M1_ex": -1, "M1": 1}
    assert mi.priors["r1_Keq"].target_id == "Keq"
    assert (
        mi.experiments["condition_1"].measurements["metabolite"]["M1"].target_type
        == "metabolite"
    )
