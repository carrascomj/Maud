# Copyright (C) 2019 Novf Denmark.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Definitions of Maud-specific objects."""

from collections import defaultdict
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanMCMC
from numpy.linalg.linalg import LinAlgError
from pydantic import BaseModel, Field, root_validator, validator


class Compartment(BaseModel):
    """Constructor for compartment objects.

    :param id: compartment id, use a BiGG id if possible.
    :param name: compartment name.
    :param volume: compartment volume.
    """
    id: str
    name: Optional[str] = None
    volume: float = 1.0


class Metabolite(BaseModel):
    """Constructor for metabolite objects.

    :param id: metabolite id, use a BiGG id if possible.
    :param name: metabolite name.
    :param external_id: metabolite name.
    """
    id: str
    name: Optional[str] = None
    inchi_key: Optional[str] = None


class MetaboliteInCompartment(BaseModel):
    """A metabolite, in a compartment, or mic for short.

    :param id: this mic's id, usually <metabolite_id>_<compartment_id>.
    :param metabolite_id: id of this mic's metabolite
    :param compartment_id: id of this mic's compartment
    :param balanced: Does this mic have stable concentration at steady state?

    """
    id: str
    metabolite_id: str
    compartment_id: str
    name: Optional[str] = None
    balanced: Optional[bool] = None


class Modifier(BaseModel):
    """Constructor for modifier objects.

    :param mic_id: the id of the modifying metabolite-in-compartment
    :param enzyme_id: the id of the modified enzyme
    :param modifier_type: what is the modifier type, e.g.
    'allosteric_activator', 'allosteric_inhibitor', 'competitive_inhibitor'
    """
    mic_id: str
    enzyme_id: str
    modifier_type: Optional[str] = None
    allosteric: bool = None

    @validator("allosteric", pre=True, always=True)
    def default_allosteric(cls, _v, *, values, **kwargs):
        return values["modifier_type"] in [
            "allosteric_inhibitor",
            "allosteric_activator",
        ]


class Parameter(BaseModel):
    """Constructor for parameter object.

    :param id: parameter id
    :param enzyme_id: id of the enzyme associated with the parameter
    :param metabolite_id: id of the metabolite associated with the parameter if any
    """
    id: str
    enzyme_id: str
    metabolite_id: Optional[str] = None
    is_thermodynamic: bool = False


class Enzyme(BaseModel):
    """Constructor for the enzyme object.

    :param id: a string identifying the enzyme
    :param reaction_id: the id of the reaction the enzyme catalyses
    :param name: human-understandable name for the enzyme
    :param modifiers: modifiers, given as {'modifier_id': modifier_object}
    :param subunits: number of subunits in enzymes
    """
    id: str
    reaction_id: str
    name: str
    modifiers: Dict[str, List[Modifier]] = Field(default=defaultdict())
    subunits: int = 1
    allosteric: bool = None

    @validator("allosteric", pre=True, always=True)
    def default_allosteric(cls, _v, *, values, **kwargs):
        return (
            len(values["modifiers"]["allosteric_activator"]) > 0
            or len(values["modifiers"]["allosteric_inhibitor"]) > 0
        )


class Reaction(BaseModel):
    """Constructor for the reaction object.

    :param id: reaction id, use a BiGG id if possible.
    :param name: reaction name.
    :param reaction_mechanism: either "reversible_modular_rate_law", "drain", or,
    "irreversible_modular_rate_law".
    :param stoichiometry: reaction stoichiometry,
    e.g. for the reaction: 2pg <-> 3pg we have {'2pg'; -1, '3pg': 1}
    :param enzymes: Dictionary mapping enzyme ids to Enzyme objects
    :param water_stroichiometry: Reaction's stoichiometric coefficient for water
    """
    id: str
    name: Optional[str] = None
    # TODO: enum might be better
    reaction_mechanism: str
    stoichiometry: Optional[Dict[str, float]] = Field(default=defaultdict())
    enzymes: Optional[List[Enzyme]] = None
    water_stoichiometry: float = 0

    @validator("name", pre=True, always=True)
    def default_name_is_id(cls, v, *, values, **kwargs):
        return v or values["id"]

    @validator("reaction_mechanism")
    def validate_reaction_mechanism(cls, v, field):
        assert v in [
            "reversible_modular_rate_law", "drain", "irreversible_modular_rate_law"
        ], "must be one of the three supported reaction mechanisms"
        return v


class Phosphorylation(BaseModel):
    """Constructor for the phosphorylation object.

    :param id: phosphorylation id. use BIGG id if possible.
    :param name: name of phosphorylation reaction.
    :param activating: if the interaction activates the
    target enzyme.
    :param inhibiting: if the interaction inhibits the
    target enzyme.
    :enzyme_id: the target enzyme of the interaction
    """
    id: str
    enzyme_id: str
    name: Optional[str] = None
    # bools are None by default
    activating: bool = None
    inhibiting: bool = None


class KineticModel(BaseModel):
    """Constructor for representation of a system of metabolic reactions.

    :param model_id: id of the kinetic model
    :param metabolites: list of metabolite objects
    :param reactions: list of reaction objects
    :param compartments: list of compartment objects
    :param mic: list of MetaboliteInCompartment objects
    """
    model_id: str
    metabolites: List[Metabolite]
    reactions: List[Reaction]
    compartments: List[Compartment]
    mics: List[MetaboliteInCompartment]
    phosphorylation: Optional[List[Phosphorylation]] = None


class MeasurementSet(BaseModel):
    """A container for a complete set of measurements, including knockouts."""

    yconc: pd.DataFrame
    yflux: pd.DataFrame
    yenz: pd.DataFrame
    enz_knockouts: pd.DataFrame
    phos_knockouts: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True


class Experiment(BaseModel):
    """Constructor for Experiment object.

    :param id: id for each experiment
    :param sample: if the experiment will be used in parameter sampling
    :param predict: if the experiment will be used in predictive samplig
    """
    id: str
    sample: bool
    predict: bool


class IndPrior1d(BaseModel):
    """Independent location/scale prior for a 1-dimentional parameter."""

    parameter_name: str
    location: pd.Series
    scale: pd.Series

    @root_validator
    def root_validator(cls, values):
        if not values["location"].index.equals(values["scale"].index):
            raise ValueError("Location index doesn't match scale index.")
        return values

    class Config:
        arbitrary_types_allowed = True


class IndPrior2d(BaseModel):
    """Independent location/scale prior for a 2-dimensional parameter."""

    parameter_name: str
    location: pd.DataFrame
    scale: pd.DataFrame

    @root_validator
    def root_validator(cls, values):
        if not values["location"].index.equals(values["scale"].index):
            raise ValueError("Location index doesn't match scale index.")
        if not values["location"].columns.equals(values["scale"].columns):
            raise ValueError("Location columns don't match scale columns.")
        return values

    class Config:
        arbitrary_types_allowed = True


class MultiVariateNormalPrior1d(BaseModel):
    """a location vector and covariance matrix prior for a 1-dimensional parameter."""

    parameter_name: str
    location: pd.Series
    covariance_matrix: pd.DataFrame

    @root_validator
    def root_validator(cls, values):
        if not values["location"].index.equals(values["covariance_matrix"].index):
            raise ValueError("Location index doesn't match scale index.")
        if not values["location"].index.equals(values["covariance_matrix"].columns):
            raise ValueError("Location index doesn't match scale columns.")
        try:
            np.linalg.cholesky(values["covariance_matrix"].values)
        except LinAlgError as e:
            raise ValueError("Covariance matrix is not positive definite") from e
        return values

    class Config:
        arbitrary_types_allowed = True


class PriorSet(BaseModel):
    """Object containing all priors for a MaudInput."""

    priors_kcat: IndPrior1d
    priors_kcat_phos: IndPrior1d
    priors_km: IndPrior1d
    priors_dgf: MultiVariateNormalPrior1d
    priors_ki: IndPrior1d
    priors_diss_t: IndPrior1d
    priors_diss_r: IndPrior1d
    priors_transfer_constant: IndPrior1d
    priors_conc_unbalanced: IndPrior2d
    priors_drain: IndPrior2d
    priors_conc_enzyme: IndPrior2d
    priors_conc_phos: IndPrior2d


class StanCoordSet(BaseModel):
    """Object containing human-readable indexes for Maud's parameters.

    These are "coordinates" in the sense of xarray

    """

    metabolites: List[str]
    mics: List[str]
    balanced_mics: List[str]
    unbalanced_mics: List[str]
    km_enzs: List[str]
    km_mics: List[str]
    reactions: List[str]
    experiments: List[str]
    enzymes: List[str]
    edges: List[str]
    allosteric_enzymes: List[str]
    drains: List[str]
    phos_enzs: List[str]
    yconc_exps: List[str]
    yconc_mics: List[str]
    yflux_exps: List[str]
    yflux_rxns: List[str]
    yenz_exps: List[str]
    yenz_enzs: List[str]
    ci_enzs: List[str]
    ci_mics: List[str]
    ai_enzs: List[str]
    ai_mics: List[str]
    aa_enzs: List[str]
    aa_mics: List[str]
    enz_ko_exps: List[str]
    enz_ko_enzs: List[str]
    phos_ko_exps: List[str]
    phos_ko_enzs: List[str]


class MaudConfig(BaseModel):
    """User's configuration for a Maud input.

    :param name: name for the input. Used to name the output directory
    :param kinetic_model_file: path to a valid kientic model file.
    :param priors_file: path to a valid priors file.
    :param experiments_file: path to a valid experiments file.
    :param likelihood: Whether or not to take measurements into account.
    :param reject_non_steady: Reject draws if a non-steady state is encountered.
    :param ode_config: Configuration for Stan's ode solver.
    :param cmdstanpy_config: Arguments to cmdstanpy.CmdStanModel.sample.
    :param stanc_options: Valid choices for CmdStanModel argument `stanc_options`.
    :param cpp_options: Valid choices for CmdStanModel `cpp_options`.
    :param variational_options: Arguments for CmdStanModel.variational.
    :param user_inits_file: path to a csv file of initial values.
    :param dgf_mean_file: path to a csv file of formation energy means.
    :param dgf_covariance_file: path to a csv file of formation energy covariances.
    :param steady_state_threshold_abs: absolute threshold for Sv=0 be at steady state
    :param steady_state_threshold_rel: relative threshold for Sv=0 be at steady state
    """

    name: str
    kinetic_model_file: str
    priors_file: str
    measurements_file: str
    biological_config_file: str
    likelihood: bool
    reject_non_steady: bool
    ode_config: dict
    cmdstanpy_config: dict
    stanc_options: Optional[dict]
    cpp_options: Optional[dict]
    variational_options: Optional[dict]
    user_inits_file: Optional[str]
    dgf_mean_file: Optional[str]
    dgf_covariance_file: Optional[str]
    steady_state_threshold_abs: float
    steady_state_threshold_rel: float


class MaudInput(BaseModel):
    """Everything that is needed to run Maud.

    :param kinetic_system: a KineticSystem object
    :param priors: a dictionary mapping prior types to lists of Prior objects
    :param stan_coords: a StanCoordSet object
    :param measurement_set: a list of Measurement objects
    :param inits: a dictionary of initial parameter values
    """
    config: MaudConfig
    kinetic_model: KineticModel
    priors: PriorSet
    stan_coords: StanCoordSet
    measurements: MeasurementSet
    all_experiments: List[Experiment]
    inits: Dict[str, Union[np.ndarray, pd.Series, pd.DataFrame]]

    @root_validator
    def root_validator(cls, values):
        kinetic_model, priors_km = (
            values.get("kinetic_model"),
            values.get("priors").priors_km,
        )
        reactions = kinetic_model.reactions
        for reaction in reactions:
            enz_index = [enz.id for enz in reaction.enzymes]
            mic_index = list(reaction.stoichiometry.keys())
            multi_indexer = (enz_index, mic_index)
            assert (
                not priors_km.location.loc[multi_indexer].isna().any()
            ), f"reaction {reaction.id} has badformed or missing location Km"
            assert (
                not priors_km.scale.loc[multi_indexer].isna().any()
            ), f"reaction {reaction.id} has badformed or missing scale Km"
        return values

    class Config:
        arbitrary_types_allowed = True


class SimulationStudyOutput:
    """Expected output of a simulation study.

    :param input_data_sim: dictionary used to create simulation
    :param input_data_sample: dictionary used to create samples
    :param true_values: dictionary mapping param names to true values
    :param sim: CmdStanMCMC that generated simulated measurements
    :param mi: Maud input used for sampling
    :param samples: CmdStanMCMC output of the simulation study
    """

    def __init__(
        self,
        input_data_sim: dict,
        input_data_sample: dict,
        true_values: dict,
        sim: CmdStanMCMC,
        mi: MaudInput,
        samples: CmdStanMCMC,
    ):
        self.input_data_sim = input_data_sim
        self.input_data_sample = input_data_sample
        self.true_values = true_values
        self.sim = sim
        self.mi = mi
        self.samples = samples
