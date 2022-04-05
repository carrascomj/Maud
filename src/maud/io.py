# Copyright (C) 2019 Novo Nordisk Foundation Center for Biosustainability,
# Technical University of Denmark.
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

"""Functions for loading MaudInput objects.

(and at some point in the future also saving MaudOutput objects)

"""

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import toml

from maud import validation
from maud.data_model import (
    Compartment,
    Enzyme,
    Experiment,
    IndPrior1d,
    IndPrior2d,
    KineticModel,
    MaudConfig,
    MaudInput,
    MeasurementSet,
    Metabolite,
    MetaboliteInCompartment,
    Modifier,
    MultiVariateNormalPrior1d,
    Phosphorylation,
    PriorSet,
    Reaction,
    StanCoordSet,
)
from maud.utils import (
    get_lognormal_parameters_from_quantiles,
    get_normal_parameters_from_quantiles,
)


DEFAULT_PRIOR_LOC_UNBALANCED = 0.1
DEFAULT_PRIOR_SCALE_UNBALANCED = 2.0
DEFAULT_PRIOR_LOC_ENZYME = 0.1
DEFAULT_PRIOR_SCALE_ENZYME = 2.0
DEFAULT_STEADY_STATE_THRESHOLD_ABS = 1e-8
DEFAULT_STEADY_STATE_THRESHOLD_REL = 1e-3
NON_LN_SCALE_PARAMS = ["dgf", "drain"]
DEFAULT_ODE_CONFIG = {
    "rel_tol": 1e-9,
    "abs_tol": 1e-9,
    "rel_tol_forward": 1e-9,
    "abs_tol_forward": 1e-9,
    "rel_tol_backward": 1e-9,
    "abs_tol_backward": 1e-9,
    "rel_tol_quadrature": 1e-9,
    "abs_tol_quadrature": 1e-9,
    "max_num_steps": int(1e9),
    "num_steps_between_checkpoints": 150,
    "interpolation_polynomial": 1,  # Hermite or change to 2 for polynomial
    "solver_forward": 2,  # BDF or change to 1 for adams
    "solver_backward": 2,  # BDF or change to 1 for adams
    "timepoint": 500,
}


def load_maud_input(data_path: str, mode: str) -> MaudInput:
    """
    Load an MaudInput object from a data path.

    :param filepath: path to directory containing input toml file
    :param mode: determines which experiments will be included,
    defined in the `biological_config` as "sample" and/or "predict".

    """
    config = parse_config(toml.load(os.path.join(data_path, "config.toml")))
    kinetic_model_path = os.path.join(data_path, config.kinetic_model_file)
    biological_config_path = os.path.join(data_path, config.biological_config_file)
    raw_kinetic_model = dict(toml.load(kinetic_model_path))
    raw_biological_config = dict(toml.load(biological_config_path))
    measurements_path = os.path.join(data_path, config.measurements_file)
    priors_path = os.path.join(data_path, config.priors_file)
    dgf_prior_paths = {
        "loc": os.path.join(data_path, str(config.dgf_mean_file)),
        "cov": os.path.join(data_path, str(config.dgf_covariance_file)),
    }
    all_experiments = get_all_experiment_object(raw_biological_config)
    kinetic_model = parse_toml_kinetic_model(raw_kinetic_model)
    raw_measurements = pd.read_csv(measurements_path)
    raw_priors = pd.read_csv(priors_path)
    assert isinstance(raw_measurements, pd.DataFrame)
    assert isinstance(raw_priors, pd.DataFrame)
    stan_coords = get_stan_coords(
        kinetic_model, raw_measurements, all_experiments, mode
    )
    measurement_set = parse_measurements(raw_measurements, stan_coords)
    if os.path.exists(dgf_prior_paths["loc"]):
        dgf_prior_loc_df = pd.read_csv(
            dgf_prior_paths["loc"], index_col="metabolite", squeeze=False
        )
        assert isinstance(dgf_prior_loc_df, pd.DataFrame)
        dgf_loc = pd.Series(dgf_prior_loc_df["prior_mean_dgf"])
    else:
        dgf_loc = pd.Series([])
    if os.path.exists(dgf_prior_paths["cov"]):
        dgf_cov = pd.DataFrame(
            pd.read_csv(dgf_prior_paths["cov"], index_col="metabolite")
        )
    else:
        dgf_cov = pd.DataFrame([])
    prior_set = parse_priors(raw_priors, stan_coords, dgf_loc, dgf_cov)

    if config.user_inits_file is not None:
        user_inits_path = os.path.join(data_path, config.user_inits_file)
    else:
        user_inits_path = None
    inits = get_inits(prior_set, user_inits_path)
    mi = MaudInput(
        config=config,
        kinetic_model=kinetic_model,
        priors=prior_set,
        stan_coords=stan_coords,
        measurements=measurement_set,
        all_experiments=all_experiments,
        inits=inits,
    )
    validation.validate_maud_input(mi)
    return mi


def parse_toml_kinetic_model(raw: dict) -> KineticModel:
    """Turn the output of toml.load into a KineticModel object.

    :param raw: Result of running toml.load on a suitable toml file

    """
    model_id = raw["model_id"] if "model_id" in raw.keys() else "mi"
    compartments = [
        Compartment(id=c["id"], name=c["name"], volume=c["volume"])
        for c in raw["compartment"]
    ]
    metabolites = []
    for r in raw["metabolite-in-compartment"]:
        if not any(r["metabolite"] == m.id for m in metabolites):
            inchi_key = (
                r["metabolite_inchi_key"]
                if "metabolite_inchi_key" in r.keys()
                else None
            )
            metabolites.append(Metabolite(id=r["metabolite"], inchi_key=inchi_key))
    mics = [
        MetaboliteInCompartment(
            id=f"{m['metabolite']}_{m['compartment']}",
            metabolite_id=m["metabolite"],
            compartment_id=m["compartment"],
            balanced=m["balanced"],
        )
        for m in raw["metabolite-in-compartment"]
    ]
    reactions = [parse_toml_reaction(r) for r in raw["reaction"]]
    if "drain" in raw.keys():
        reactions += [parse_toml_drain(d) for d in raw["drain"]]
    phosphorylation = (
        [parse_toml_phosphorylation(d) for d in raw["phosphorylation"]]
        if "phosphorylation" in raw.keys()
        else []
    )
    return KineticModel(
        model_id=model_id,
        metabolites=metabolites,
        compartments=compartments,
        mics=mics,
        reactions=reactions,
        phosphorylation=phosphorylation,
    )


def get_km_coords(rxns: List[Reaction]) -> List[Tuple[str]]:
    """Get coordinates for km parameters.

    The returned coordinate set for kms is dependent on if it's reversible.
    :param rxns: Reaction object
    """
    coords = []
    for r in rxns:
        for e in r.enzymes:
            if r.reaction_mechanism == "irreversible_modular_rate_law":
                coords += [(e.id, m[0]) for m in r.stoichiometry.items() if m[1] < 0]
            elif r.reaction_mechanism == "reversible_modular_rate_law":
                coords += [(e.id, m[0]) for m in r.stoichiometry.items()]
    return sorted(coords)


def get_stan_coords(
    km: KineticModel,
    raw_measurements: pd.DataFrame,
    all_experiments: List[Experiment],
    mode: str,
) -> StanCoordSet:
    """Get all the coordinates that Maud needs.

    This function is responsible for setting the order of each parameter.

    :param km: KineticModel object
    :param raw_measurements: MeasurementSet object
    :param all_experiments: List of Experiment object,
    all experiments in biological config file
    :param mode: "sample" or "predict" determine which experiments will be used
    """

    def unpack(packed):
        return map(list, zip(*packed)) if len(packed) > 0 else ([], [])

    measurement_types = [
        "mic",
        "flux",
        "enzyme",
        "knockout_enz",
        "knockout_phos",
    ]
    modifier_types = [
        "competitive_inhibitor",
        "allosteric_inhibitor",
        "allosteric_activator",
    ]
    metabolites = sorted([m.id for m in km.metabolites])
    mics = sorted([m.id for m in km.mics])
    balanced_mics = sorted([m.id for m in km.mics if m.balanced])
    unbalanced_mics = sorted([m.id for m in km.mics if not m.balanced])
    reactions = sorted([r.id for r in km.reactions])
    enzymes = sorted([e.id for r in km.reactions for e in r.enzymes])
    allosteric_enzymes = sorted(
        [e.id for r in km.reactions for e in r.enzymes if e.allosteric]
    )
    phos_enzs = sorted([e.id for e in km.phosphorylation])
    drains = sorted([r.id for r in km.reactions if r.reaction_mechanism == "drain"])
    edges = enzymes + drains
    experiments = sorted([e.id for e in all_experiments if getattr(e, mode)])
    pmf = experiments.copy()
    ci_coords, ai_coords, aa_coords = (
        sorted(
            [
                (m.enzyme_id, m.mic_id)
                for r in km.reactions
                for e in r.enzymes
                for m in e.modifiers[modifier_type]
            ]
        )
        for modifier_type in modifier_types
    )
    ci_enzs, ci_mics = unpack(ci_coords)
    ai_enzs, ai_mics = unpack(ai_coords)
    aa_enzs, aa_mics = unpack(aa_coords)
    km_coords = get_km_coords(km.reactions)
    km_enzs, km_mics = map(list, zip(*km_coords))
    yc_coords, yf_coords, ye_coords, enz_ko_coords, phos_ko_coords = (
        sorted(
            [
                (row["experiment_id"], row["target_id"])
                for _, row in raw_measurements.iterrows()
                if (row["measurement_type"] == measurement_type)
                and (row["experiment_id"] in experiments)
            ]
        )
        for measurement_type in measurement_types
    )
    yconc_exps, yconc_mics = unpack(yc_coords)
    yflux_exps, yflux_rxns = unpack(yf_coords)
    yenz_exps, yenz_enzs = unpack(ye_coords)
    enz_ko_exps, enz_ko_enzs = unpack(enz_ko_coords)
    phos_ko_exps, phos_ko_enzs = unpack(phos_ko_coords)
    return StanCoordSet(
        metabolites=metabolites,
        mics=mics,
        km_enzs=km_enzs,
        km_mics=km_mics,
        balanced_mics=balanced_mics,
        unbalanced_mics=unbalanced_mics,
        reactions=reactions,
        edges=edges,
        experiments=experiments,
        enzymes=enzymes,
        allosteric_enzymes=allosteric_enzymes,
        phos_enzs=phos_enzs,
        drains=drains,
        yconc_exps=yconc_exps,
        yconc_mics=yconc_mics,
        yflux_exps=yflux_exps,
        yflux_rxns=yflux_rxns,
        yenz_exps=yenz_exps,
        yenz_enzs=yenz_enzs,
        ci_enzs=ci_enzs,
        ci_mics=ci_mics,
        ai_enzs=ai_enzs,
        ai_mics=ai_mics,
        aa_enzs=aa_enzs,
        aa_mics=aa_mics,
        enz_ko_exps=enz_ko_exps,
        enz_ko_enzs=enz_ko_enzs,
        phos_ko_exps=phos_ko_exps,
        phos_ko_enzs=phos_ko_enzs,
        pmf=pmf,
    )


def parse_toml_drain(raw: dict) -> Reaction:
    """Turn a dictionary representing a drain into a Reaction object.

    :param raw: Dictionary representing a drain, typically one of
    the values of the 'drains' field in the output of toml.load.

    """

    return Reaction(
        id=raw["id"],
        name=raw["name"],
        reaction_mechanism="drain",
        stoichiometry=raw["stoichiometry"],
        enzymes=[],
    )


def parse_toml_phosphorylation(raw: dict) -> Phosphorylation:
    """Turn a dictionary-format phosphorylation into a Maud object.

    :param toml_phosphorylation: Dictionary representing a phosphorylation
    enzyme, typically one of the values of the 'phosphorylation' field in the
    output of toml.load.

    """

    return Phosphorylation(
        id=raw["id"],
        name=raw["name"],
        activating=raw["activating"] if "activating" in raw.keys() else None,
        inhibiting=raw["inhibiting"] if "inhibiting" in raw.keys() else None,
        enzyme_id=raw["enzyme_id"],
    )


def parse_toml_reaction(raw: dict) -> Reaction:
    """Turn a dictionary representing a reaction into a Reaction object.

    :param raw: Dictionary representing a reaction, typically one of
    the values of the 'reactions' field in the output of toml.load.

    """
    enzymes = []
    subunits = 1
    water_stoichiometry = (
        raw["water_stoichiometry"] if "water_stoichiometry" in raw.keys() else 0
    )
    transported_charge = (
        raw["transported_charge"] if "transported_charge" in raw else 0
    )
    for e in raw["enzyme"]:
        modifiers = {
            "competitive_inhibitor": [],
            "allosteric_inhibitor": [],
            "allosteric_activator": [],
        }
        if "modifier" in e.keys():
            for modifier_dict in e["modifier"]:
                modifier_type = modifier_dict["modifier_type"]
                modifiers[modifier_type].append(
                    Modifier(
                        mic_id=modifier_dict["mic_id"],
                        enzyme_id=e["id"],
                        modifier_type=modifier_type,
                    )
                )

        if "subunit" in e.keys():
            subunits = e["subunit"]

        enzymes.append(
            Enzyme(
                id=e["id"],
                name=e["name"],
                reaction_id=raw["id"],
                modifiers=modifiers,
                subunits=subunits,
            )
        )
    return Reaction(
        id=raw["id"],
        name=raw["name"],
        reaction_mechanism=raw["mechanism"],
        stoichiometry=raw["stoichiometry"],
        enzymes=enzymes,
        water_stoichiometry=water_stoichiometry,
        transported_charge=transported_charge,
    )


def parse_measurements(raw: pd.DataFrame, cs: StanCoordSet) -> MeasurementSet:
    """Parse a measurements dataframe.

    :param raw: result of running pd.read_csv on a suitable file
    """
    type_to_coords = {
        "mic": [cs.yconc_exps, cs.yconc_mics],
        "flux": [cs.yflux_exps, cs.yflux_rxns],
        "enzyme": [cs.yenz_exps, cs.yenz_enzs],
        "knockout_enz": [cs.enz_ko_exps, cs.enz_ko_enzs],
        "knockout_phos": [cs.phos_ko_exps, cs.phos_ko_enzs],
    }
    yconc, yflux, yenz = (
        raw.loc[lambda df: df["measurement_type"] == t]
        .set_index(["experiment_id", "target_id"])
        .reindex(
            pd.MultiIndex.from_arrays(
                type_to_coords[t], names=["experiment_id", "target_id"]
            )
        )
        for t in ["mic", "flux", "enzyme"]
    )
    enz_knockouts = pd.DataFrame(False, index=cs.experiments, columns=cs.enzymes)
    phos_knockouts = pd.DataFrame(False, index=cs.experiments, columns=cs.phos_enzs)
    for _, row in raw.loc[
        lambda df: df["measurement_type"] == "knockout_enz"
    ].iterrows():
        if row["experiment_id"] in cs.experiments:
            enz_knockouts.loc[row["experiment_id"], row["target_id"]] = True
    for _, row in raw.loc[
        lambda df: df["measurement_type"] == "knockout_phos"
    ].iterrows():
        if row["experiment_id"] in cs.experiments:
            phos_knockouts.loc[row["experiment_id"], row["target_id"]] = True
    return MeasurementSet(
        yconc=yconc,
        yflux=yflux,
        yenz=yenz,
        enz_knockouts=enz_knockouts,
        phos_knockouts=phos_knockouts,
    )


def get_all_experiment_object(
    raw: dict,
) -> List[Experiment]:
    """Get experiments from biological_context toml.

    :param raw: a dictionary the comes from a biological_context toml.
    """

    return [
        Experiment(id=exp["id"], sample=exp["sample"], predict=exp["predict"])
        for exp in raw["experiment"]
    ]


def extract_1d_prior(
    raw: pd.DataFrame,
    parameter_name: str,
    coords: List[List[str]],
    default_loc: float = np.nan,
    default_scale: float = np.nan,
) -> IndPrior1d:
    """Get an IndPrior1d object from a csv, with optional defaults.

    :param raw: a pandas dataframe that comes from a priors csv

    :param parameter_name: name of the parameter being extracted. Must match
    the parameter_name column of the csv

    :param coords: a list of id columns that can be used to identify the rows
    in the csv.

    :param default_loc: default value for prior location.

    :param default_scale: default value for prior scale.

    """
    parameter_name_to_id_cols = {
        "kcat": ["enzyme_id"],
        "km": ["enzyme_id", "mic_id"],
        "dgf": ["metabolite_id"],
        "diss_t": ["enzyme_id", "mic_id"],
        "diss_r": ["enzyme_id", "mic_id"],
        "ki": ["enzyme_id", "mic_id"],
        "transfer_constant": ["enzyme_id"],
        "kcat_phos": ["phos_enz_id"],
        "pmf": ["experiment_id"],
    }
    if any(len(c) == 0 for c in coords):
        return IndPrior1d(
            parameter_name=parameter_name,
            location=pd.Series([], dtype="float64"),
            scale=pd.Series([], dtype="float64"),
        )
    non_negative = parameter_name not in ["dgf"]
    qfunc = (
        get_lognormal_parameters_from_quantiles
        if non_negative
        else get_normal_parameters_from_quantiles
    )
    id_cols = parameter_name_to_id_cols[parameter_name]
    if len(id_cols) == 1:
        ix = pd.Index(coords[0], name=id_cols[0])
    else:
        ix = pd.MultiIndex.from_arrays(coords, names=id_cols)
    out = (
        raw.loc[lambda df: df["parameter_type"] == parameter_name]
        .set_index(id_cols)
        .reindex(ix)
    )
    pct_params = (
        out[["pct1", "pct99"]]
        .apply(
            lambda row: qfunc(row["pct1"], 0.01, row["pct99"], 0.99),
            axis=1,
            result_type="expand",
        )
        .rename(columns={0: "location", 1: "scale"})
    )
    pct_params["location"] = np.exp(pct_params["location"])
    out[["location", "scale"]] = out[["location", "scale"]].fillna(pct_params)
    return IndPrior1d(
        parameter_name=parameter_name,
        location=out["location"].fillna(default_loc),
        scale=out["scale"].fillna(default_scale),
    )


def extract_2d_prior(
    raw: pd.DataFrame,
    parameter_name: str,
    row_coords: List[str],
    col_coords: List[str],
    default_loc: float = np.nan,
    default_scale: float = np.nan,
) -> IndPrior2d:
    """Get an IndPrior2d object from a csv, with optional defaults.

    :param raw: a pandas dataframe that comes from a priors csv

    :param parameter_name: name of the parameter being extracted. Must match
    the parameter_name column of the csv

    :param row_coords: a list of strings corresponding to the rows of the
    output, usually experiment_id.

    :param col_coords: a list of strings corresponding to the columns of the
    output.

    :param default_loc: default value for prior location.

    :param default_scale: default value for prior scale.

    """
    parameter_name_to_id_cols = {
        "conc_unbalanced": ["experiment_id", "mic_id"],
        "conc_enzyme": ["experiment_id", "enzyme_id"],
        "drain": ["experiment_id", "drain_id"],
        "conc_phos": ["experiment_id", "phos_enz_id"],
    }
    if len(col_coords) == 0:
        return IndPrior2d(
            parameter_name=parameter_name,
            location=pd.DataFrame([]),
            scale=pd.DataFrame([]),
        )
    non_negative = parameter_name not in ["drain"]
    qfunc = (
        get_lognormal_parameters_from_quantiles
        if non_negative
        else get_normal_parameters_from_quantiles
    )
    out = raw.loc[lambda df: df["parameter_type"] == parameter_name].set_index(
        parameter_name_to_id_cols[parameter_name]
    )
    pct_params = out[["pct1", "pct99"]].apply(
        lambda row: qfunc(row["pct1"], 0.01, row["pct99"], 0.99),
        axis=1,
        result_type="expand",
    )
    pct_params.columns = ["location", "scale"]
    pct_params["location"] = np.exp(pct_params["location"])
    out[["location", "scale"]] = out[["location", "scale"]].fillna(pct_params)
    location, scale = (
        out[col]
        .unstack()
        .reindex(row_coords, columns=col_coords)
        .rename_axis(parameter_name_to_id_cols[parameter_name][0])
        .rename_axis(parameter_name_to_id_cols[parameter_name][1], axis="columns")
        for col in ["location", "scale"]
    )
    return IndPrior2d(
        parameter_name=parameter_name,
        location=location.fillna(default_loc),
        scale=scale.fillna(default_scale),
    )


def parse_priors(
    raw: pd.DataFrame,
    cs: StanCoordSet,
    dgf_loc: pd.Series,
    dgf_cov: pd.DataFrame,
) -> PriorSet:
    """Get a PriorSet object from a dataframe of raw priors.

    :param raw: result of running pd.read_csv on a suitable file
    """
    met_ix = pd.Index(cs.metabolites, name="metabolite")
    if len(dgf_loc) > 0 and len(dgf_cov) > 0:
        loc = pd.Series(dgf_loc.reindex(met_ix))
        cov = dgf_cov.reindex(met_ix, axis=0).reindex(met_ix, axis=1)
    else:
        dgf_prior_1d = extract_1d_prior(raw, "dgf", [cs.metabolites])
        loc = dgf_prior_1d.location
        cov = pd.DataFrame(np.diag(dgf_prior_1d.scale), index=met_ix, columns=met_ix)
    priors_dgf = MultiVariateNormalPrior1d(
        parameter_name="dgf", location=loc, covariance_matrix=cov
    )
    return PriorSet(
        # 1d priors
        priors_kcat=extract_1d_prior(raw, "kcat", [cs.enzymes]),
        priors_km=extract_1d_prior(raw, "km", [cs.km_enzs, cs.km_mics]),
        priors_dgf=priors_dgf,
        priors_diss_t=extract_1d_prior(raw, "diss_t", [cs.ai_enzs, cs.ai_mics]),
        priors_diss_r=extract_1d_prior(raw, "diss_r", [cs.aa_enzs, cs.aa_mics]),
        priors_ki=extract_1d_prior(raw, "ki", [cs.ci_enzs, cs.ci_mics]),
        priors_transfer_constant=extract_1d_prior(
            raw, "transfer_constant", [cs.allosteric_enzymes]
        ),
        priors_kcat_phos=extract_1d_prior(raw, "kcat_phos", [cs.phos_enzs]),
        priors_pmf=extract_1d_prior(raw, "pmf", [cs.pmf]),
        # 2d priors
        priors_drain=extract_2d_prior(raw, "drain", cs.experiments, cs.drains),
        priors_conc_enzyme=extract_2d_prior(
            raw,
            "conc_enzyme",
            cs.experiments,
            cs.enzymes,
            default_loc=DEFAULT_PRIOR_LOC_ENZYME,
            default_scale=DEFAULT_PRIOR_SCALE_ENZYME,
        ),
        priors_conc_unbalanced=extract_2d_prior(
            raw,
            "conc_unbalanced",
            cs.experiments,
            cs.unbalanced_mics,
            default_loc=DEFAULT_PRIOR_LOC_UNBALANCED,
            default_scale=DEFAULT_PRIOR_SCALE_UNBALANCED,
        ),
        priors_conc_phos=extract_2d_prior(
            raw, "conc_phos", cs.experiments, cs.phos_enzs
        ),
    )


def parse_config(raw):
    """Get a MaudConfig object from the result of toml.load.

    :param raw: result of running toml.load on a suitable file
    """
    user_inits_file: Optional[str] = (
        raw["user_inits_file"] if "user_inits_file" in raw.keys() else None
    )
    dgf_mean_file: Optional[str] = (
        raw["dgf_mean_file"] if "dgf_mean_file" in raw.keys() else None
    )
    dgf_covariance_file: Optional[str] = (
        raw["dgf_covariance_file"] if "dgf_covariance_file" in raw.keys() else None
    )
    reject_non_steady: bool = (
        raw["reject_non_steady"] if "reject_non_steady" in raw.keys() else True
    )
    stanc_options: Optional[dict] = (
        raw["stanc_options"] if "stanc_options" in raw.keys() else None
    )
    cpp_options: Optional[dict] = (
        raw["cpp_options"] if "cpp_options" in raw.keys() else None
    )
    variational_options: Optional[dict] = (
        raw["variational_options"] if "variational_options" in raw.keys() else None
    )
    steady_state_threshold_abs: float = (
        raw["steady_state_threshold_abs"]
        if "steady_state_threshold_abs" in raw.keys()
        else DEFAULT_STEADY_STATE_THRESHOLD_ABS
    )
    steady_state_threshold_rel: float = (
        raw["steady_state_threshold_abs"]
        if "steady_state_threshold_abs" in raw.keys()
        else DEFAULT_STEADY_STATE_THRESHOLD_REL
    )

    return MaudConfig(
        name=raw["name"],
        kinetic_model_file=raw["kinetic_model"],
        priors_file=raw["priors"],
        measurements_file=raw["measurements"],
        biological_config_file=raw["biological_config"],
        likelihood=raw["likelihood"],
        reject_non_steady=reject_non_steady,
        ode_config={**DEFAULT_ODE_CONFIG, **raw["ode_config"]},
        cmdstanpy_config=raw["cmdstanpy_config"],
        stanc_options=stanc_options,
        cpp_options=cpp_options,
        variational_options=variational_options,
        user_inits_file=user_inits_file,
        dgf_mean_file=dgf_mean_file,
        dgf_covariance_file=dgf_covariance_file,
        steady_state_threshold_abs=steady_state_threshold_abs,
        steady_state_threshold_rel=steady_state_threshold_rel,
    )


def get_inits(priors: PriorSet, user_inits_path) -> Dict[str, np.ndarray]:
    """Get a dictionary of initial values.

    :param priors: Priorset object

    :param user_inits_path: path to a csv of user-specified initial parameter
    values

    """

    def user_inits_for_param(
        u: pd.DataFrame, p: Union[IndPrior1d, IndPrior2d]
    ) -> pd.Series:
        if len(p.location) == 0:
            return pd.Series(p.location)
        elif isinstance(p, IndPrior1d) or isinstance(p, MultiVariateNormalPrior1d):
            return u.loc[lambda df: df["parameter_name"] == p.parameter_name].set_index(
                p.location.index.names
            )["value"]
        elif isinstance(p, IndPrior2d):
            return u.loc[lambda df: df["parameter_name"] == p.parameter_name].set_index(
                [p.location.index.name, p.location.columns.name]
            )["value"]
        else:
            raise ValueError("Unrecognised prior type: " + str(type(p)))

    inits = {p.parameter_name: p.location for p in priors.__dict__.values()}
    if user_inits_path is not None:
        user_inits_all = pd.read_csv(user_inits_path)
        assert isinstance(user_inits_all, pd.DataFrame)
        for p in priors.__dict__.values():
            if p.location.empty:
                continue
            user = user_inits_for_param(user_inits_all, p)
            default = inits[p.parameter_name]
            if isinstance(inits[p.parameter_name], pd.DataFrame):
                default = default.stack()
            combined = pd.Series(
                np.where(
                    user.reindex(default.index).notnull(),
                    user.reindex(default.index),
                    default,
                ),
                index=default.index,
            )
            if isinstance(inits[p.parameter_name], pd.DataFrame):
                combined = combined.unstack()
            inits[p.parameter_name] = combined
    return rescale_inits(inits, priors)


def rescale_inits(inits: dict, priors: PriorSet) -> Dict[str, np.ndarray]:
    """Augment a dictionary of inits with equivalent normalised values.

    :param inits: original inits

    :param priors: PriorSet object used to do the normalising
    """
    rescaled = {}
    for (n, i), prior in zip(inits.items(), priors.__dict__.values()):
        if isinstance(prior, MultiVariateNormalPrior1d):
            continue
        elif n in NON_LN_SCALE_PARAMS:
            rescaled[n + "_z"] = (i - prior.location) / prior.scale
        else:
            rescaled[f"log_{n}_z"] = (np.log(i) - np.log(prior.location)) / prior.scale
    return {**inits, **rescaled}
