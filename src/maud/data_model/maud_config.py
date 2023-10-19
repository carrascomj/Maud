"""Provides dataclass MaudConfig."""
from typing import Optional, List

from pydantic.dataclasses import Field, dataclass


@dataclass(frozen=True)
class ODEConfig:
    """Config that is specific to the ODE solver."""

    rel_tol: float = 1e-9
    abs_tol: float = 1e-9
    max_num_steps: int = int(1e9)
    timepoint: float = 500


@dataclass
class MaudConfig:
    """User's configuration for a Maud input.

    :param name: name for the input. Used to name the output directory
    :param kinetic_model_file: path to a valid kientic model file.
    :param priors_file: path to a valid priors file.
    :param experiments_file: path to a valid experiments file.
    :param likelihood: Whether or not to take measurements into account.
    :param cmdstanpy_config: Arguments to cmdstanpy.CmdStanModel.sample.
    :param reject_non_steady: Reject draws if a non-steady state is encountered.
    :param ode_config: Configuration for Stan's ode solver.
    :param stanc_options: Valid choices for CmdStanModel argument `stanc_options`.
    :param cpp_options: Valid choices for CmdStanModel `cpp_options`.
    :param variational_options: Arguments for CmdStanModel.variational.
    :param optimize_options: Arguments for CmdStanModel.optimize.
    :param user_inits_file: path to a csv file of initial values.
    :param steady_state_threshold_abs: absolute threshold for Sv=0 be at steady state
    :param steady_state_threshold_rel: relative threshold for Sv=0 be at steady state
    :param default_initial_concentration: in molecule_unit per volume_unit
    :param drain_small_conc_corrector: number for correcting small conc drains
    :param molecule_unit: A unit for counting molecules, like 'mol' or 'mmol'
    :param volume_unit: A unit for measuring volume, like 'L'
    :param energy_unit: A unit for measuring energy, like 'J' or 'kJ'
    """

    name: str
    kinetic_model_file: str
    priors_file: str
    experiments_file: str
    likelihood: bool
    cmdstanpy_config: Optional[dict] = None
    cmdstanpy_config_predict: Optional[dict] = None
    stanc_options: Optional[dict] = None
    cpp_options: Optional[dict] = None
    variational_options: Optional[dict] = None
    optimize_options: Optional[dict] = None
    user_inits_file: Optional[str] = None
    ode_config: ODEConfig = Field(default_factory=ODEConfig)
    reject_non_steady: bool = True
    steady_state_threshold_abs: float = 1e-8
    steady_state_threshold_rel: float = 1e-3
    default_initial_concentration: float = 0.01
    drain_small_conc_corrector: float = 1e-6
    quench_timepoints: List[float] = Field(default_factory=list)
    molecule_unit: str = "mmol"
    volume_unit: str = "L"
    energy_unit: str = "kJ"
