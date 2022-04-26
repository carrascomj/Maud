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

"""Functions that are exposed to the command line interface live here."""

import os
import shutil
from datetime import datetime

import arviz as az
import click

from maud.getting_idatas import get_idata
from maud.loading_maud_inputs import load_maud_input
from maud.running_stan import predict, sample, simulate, variational


RELATIVE_PATH_EXAMPLE = "../../tests/data/linear"


def get_example_path(relative_path_example):
    """Get absolute path to file containing example input."""

    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, relative_path_example)


@click.group()
@click.help_option("--help", "-h")
def cli():
    """Use Maud's command line interface."""
    pass


@cli.command("sample")
@click.option("--output_dir", default=".", help="Where to save Maud's output")
@click.argument(
    "data_path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default=get_example_path(RELATIVE_PATH_EXAMPLE),
)
def sample_command(data_path, output_dir):
    """Run the sample function as a click command."""
    click.echo(do_sample(data_path, output_dir))


def do_sample(data_path, output_dir):
    """Generate MCMC samples given a user input directory.

    This function creates a new directory in output_dir with a name starting
    with "maud_output". It first copies the directory at data_path into the new
    this directory at new_dir/user_input, then runs the running_stan.sample
    function to write samples in new_dir/samples. Finally it prints the results
    of cmdstanpy's diagnose and summary methods.

    """
    mi = load_maud_input(data_path)
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    output_name = f"maud_output-{mi.config.name}-{now}"
    output_path = os.path.join(output_dir, output_name)
    samples_path = os.path.join(output_path, "samples")
    ui_dir = os.path.join(output_path, "user_input")
    print("Creating output directory: " + output_path)
    os.mkdir(output_path)
    os.mkdir(samples_path)
    print(f"Copying user input from {data_path} to {ui_dir}")
    shutil.copytree(data_path, ui_dir)
    stanfit = sample(mi, samples_path)
    print(stanfit.diagnose())
    print(stanfit.summary())
    idata = get_idata(stanfit.runset.csv_files, mi)
    idata.to_netcdf(os.path.join(output_path, "idata.nc"))
    return output_path


@cli.command("predict")
@click.argument(
    "data_path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
)
def predict_command(data_path):
    """Run the sample function as a click command."""
    click.echo(do_predict(data_path))


def do_predict(data_path: str):
    """Generate MCMC samples given a Maud output folder at train_path.

    This function creates a new directory in output_dir with a name starting
    with "maud-predict-output". It first copies the testing directory at
    train_path into the new this directory at new_dir/user_input, then runs the
    running_stan.predict_out_of_sample function to write samples in
    new_dir/oos_samples.

    The trained output is stored in the new_dir/trained_samples folder along
    with the user input required to generate the trained samples.

    """
    idata_train = az.from_netcdf(os.path.join(data_path, "idata.nc"))
    mi = load_maud_input(os.path.join(data_path, "user_input"))
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    output_name = f"maud-predict_output-{mi.config.name}-{now}"
    output_path = os.path.join(data_path, output_name)
    test_samples_path = os.path.join(output_path, "test_samples")
    print("Creating output directory: " + output_path)
    os.mkdir(output_path)
    os.mkdir(test_samples_path)
    idata_predict = predict(mi, output_path, idata_train)
    # delete attrs hack to make netcdf save work:
    # https://github.com/arviz-devs/arviz/issues/1554
    idata_predict.sample_stats.attrs = {}  # type: ignore
    idata_predict.posterior.attrs = {}  # type: ignore
    idata_predict.to_netcdf(os.path.join(output_path, "idata_predict.nc"))


@cli.command("simulate")
@click.option("--output_dir", default=".", help="Where to save the output")
@click.option("-n", default=1, type=int, help="Number of simulations")
@click.argument(
    "data_path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default=get_example_path(RELATIVE_PATH_EXAMPLE),
)
def simulate_command(data_path, output_dir, n):
    """Run the simulate function as a click command."""
    click.echo(do_simulate(data_path, output_dir, n))


def do_simulate(data_path, output_dir, n):
    """Generate draws from the prior mean."""

    mi = load_maud_input(data_path=data_path)
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    output_name = f"maud_output_sim-{mi.config.name}-{now}"
    output_path = os.path.join(output_dir, output_name)
    samples_path = os.path.join(output_path, "samples")
    ui_dir = os.path.join(output_path, "user_input")
    print("Creating output directory: " + output_path)
    os.mkdir(output_path)
    os.mkdir(samples_path)
    print(f"Copying user input from {data_path} to {ui_dir}")
    stanfit = simulate(mi, samples_path, n)
    idata = get_idata(stanfit.runset.csv_files, mi)
    idata.to_netcdf(os.path.join(output_path, "idata.nc"))
    print("\n\nSimulated concentrations:")
    print(
        idata.posterior["conc"]
        .mean(dim=["chain", "draw"])
        .to_series()
        .unstack()
        .T
    )
    print("\n\nSimulated fluxes:")
    print(
        idata.posterior["flux"]
        .mean(dim=["chain", "draw"])
        .to_series()
        .unstack()
        .T
    )
    print("\n\nSimulated enzyme concentrations:")
    print(
        idata.posterior["conc_enzyme"]
        .mean(dim=["chain", "draw"])
        .to_series()
        .unstack()
        .T
    )
    print("\n\nSimulated reaction delta Gs:")
    print(idata.posterior["dgrs"].mean(dim=["chain", "draw"]).to_series())
    print("\n\nSimulated measurements:")
    print(idata.posterior["yconc_sim"].mean(dim=["chain", "draw"]).to_series())
    print(idata.posterior["yflux_sim"].mean(dim=["chain", "draw"]).to_series())
    print("\n\nSimulated log likelihoods:")
    print(
        idata.posterior["log_lik_conc"].mean(dim=["chain", "draw"]).to_series()
    )
    print(
        idata.posterior["log_lik_flux"].mean(dim=["chain", "draw"]).to_series()
    )
    print("\n\nSimulated allostery:")
    print(
        idata.posterior["allostery"]
        .mean(dim=["chain", "draw"])
        .to_series()
        .unstack()
        .T
    )
    print("\n\nSimulated reversibilities:")
    print(
        idata.posterior["reversibility"]
        .mean(dim=["chain", "draw"])
        .to_series()
        .unstack()
        .T
    )
    print("\n\nSimulated saturation:")
    print(
        idata.posterior["saturation"]
        .mean(dim=["chain", "draw"])
        .to_series()
        .unstack()
        .T
    )
    print("\n\nSimulated membrane potential:")
    print(idata.posterior["psi"].mean(dim=["chain", "draw"]).to_series().T)
    return output_path


@cli.command("variational")
@click.option("--output_dir", default=".", help="Where to save Maud's output")
@click.argument(
    "data_path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default=get_example_path(RELATIVE_PATH_EXAMPLE),
)
def variational_command(data_path, output_dir):
    """Run the sample function as a click command."""
    click.echo(do_variational(data_path, output_dir))


def do_variational(data_path, output_dir):
    """Generate MCMC samples given a user input directory.

    This function creates a new directory in output_dir with a name starting
    with "maud_output". It first copies the directory at data_path into the new
    this directory at new_dir/user_input, then runs the running_stan.sample
    function to write samples in new_dir/samples. Finally it prints the results
    of cmdstanpy's diagnose and summary methods.

    """
    mi = load_maud_input(data_path, mode="sample")
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    output_name = f"maud_output_vi-{mi.config.name}-{now}"
    output_path = os.path.join(output_dir, output_name)
    samples_path = os.path.join(output_path, "samples")
    ui_dir = os.path.join(output_path, "user_input")
    print("Creating output directory: " + output_path)
    os.mkdir(output_path)
    os.mkdir(samples_path)
    print(f"Copying user input from {data_path} to {ui_dir}")
    shutil.copytree(data_path, ui_dir)
    variational(mi, samples_path)
    return output_path
