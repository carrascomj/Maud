#include functions.stan
data {
  // network properties
  int<lower=1> N_mic;
  int<lower=1> N_edge_sub;
  int<lower=1> N_edge_prod;
  int<lower=1> N_unbalanced;
  int<lower=1> N_metabolite;
  int<lower=1> N_km;
  int<lower=1> N_sub_km;
  int<lower=1> N_prod_km;
  int<lower=1> N_reaction;
  int<lower=1> N_enzyme;
  int<lower=0> N_drain;
  int<lower=1> N_edge;
  int<lower=0> N_allostery;
  int<lower=0> N_allosteric_enzyme;
  int<lower=0> N_phosphorylation;
  int<lower=0> N_pme;  // phosphorylation modifying enzyme
  int<lower=0> N_competitive_inhibition;
  int<lower=0> N_quench_timepoints;
  matrix[N_mic, N_edge] S;
  array[N_mic-N_unbalanced] int<lower=1,upper=N_mic> balanced_mic_ix;
  array[N_unbalanced] int<lower=1,upper=N_mic> unbalanced_mic_ix;
  array[N_competitive_inhibition] int<lower=1,upper=N_mic> ci_mic_ix;
  array[N_edge] int<lower=1,upper=3> edge_type;  // 1 = reversible modular rate law, 2 = drain
  array[N_edge] int<lower=0,upper=N_enzyme> edge_to_enzyme;  // 0 if drain
  array[N_edge] int<lower=0,upper=N_allostery> edge_to_tc;  // 0 if non-allosteric
  array[N_edge] int<lower=0,upper=N_drain> edge_to_drain;  // 0 if enzyme
  array[N_edge] int<lower=0,upper=N_reaction> edge_to_reaction;
  array[N_allostery] int<lower=1,upper=2> allostery_type;
  array[N_allostery] int<lower=1,upper=N_mic> allostery_mic;
  array[N_phosphorylation] int<lower=1,upper=2> phosphorylation_type;
  array[N_phosphorylation] int<lower=1,upper=N_pme> phosphorylation_pme;
  array[N_edge_sub] int sub_by_edge_long;
  array[N_edge,2] int sub_by_edge_bounds;
  array[N_edge_prod] int prod_by_edge_long;
  array[N_edge, 2] int prod_by_edge_bounds;
  array[N_sub_km] int sub_km_ix_by_edge_long;
  array[N_edge, 2] int sub_km_ix_by_edge_bounds;
  array[N_prod_km] int prod_km_ix_by_edge_long;
  array[N_edge,2] int prod_km_ix_by_edge_bounds;
  array[N_competitive_inhibition] int ci_ix_long;
  array[N_edge, 2] int ci_ix_bounds;
  array[N_allostery] int allostery_ix_long;
  array[N_edge, 2] int allostery_ix_bounds;
  array[N_phosphorylation] int phosphorylation_ix_long;
  array[N_edge, 2] int phosphorylation_ix_bounds;
  array[N_mic] int<lower=1,upper=N_metabolite> mic_to_met;
  vector[N_edge] water_stoichiometry;
  vector[N_edge] transported_charge;
  vector<lower=1>[N_enzyme] subunits;
  // experiment properties
  int<lower=1> N_experiment_train;
  int<lower=0> N_flux_measurement_train;
  int<lower=0> N_enzyme_measurement_train;
  int<lower=0> N_conc_measurement_train;
  int<lower=0> N_enzyme_knockout_train;
  int<lower=0> N_pme_knockout_train;
  array[N_enzyme_knockout_train] int<lower=0,upper=N_enzyme> enzyme_knockout_train_long;
  array[N_experiment_train, 2] int enzyme_knockout_train_bounds;
  array[N_pme_knockout_train] int<lower=0,upper=N_pme> pme_knockout_train_long;
  array[N_experiment_train, 2] int pme_knockout_train_bounds;
  vector[N_experiment_train] temperature_train;
  array[N_conc_measurement_train] int<lower=1,upper=N_experiment_train> experiment_yconc_train;
  array[N_conc_measurement_train] int<lower=1,upper=N_mic> mic_ix_yconc_train;
  array[N_conc_measurement_train] real yconc_train;
  vector<lower=0>[N_conc_measurement_train] sigma_yconc_train;
  array[N_flux_measurement_train] int<lower=1,upper=N_experiment_train> experiment_yflux_train;
  array[N_flux_measurement_train] int<lower=1,upper=N_reaction> reaction_yflux_train;
  array[N_flux_measurement_train] real yflux_train;
  vector<lower=0>[N_flux_measurement_train] sigma_yflux_train;
  array[N_enzyme_measurement_train] int<lower=0,upper=N_experiment_train> experiment_yenz_train;
  array[N_enzyme_measurement_train] int<lower=0,upper=N_enzyme> enzyme_yenz_train;
  array[N_enzyme_measurement_train] real yenz_train;
  vector<lower=0>[N_enzyme_measurement_train] sigma_yenz_train;
  // hardcoded priors (system)
  vector[N_metabolite] prior_loc_dgf;
  cov_matrix[N_metabolite] prior_cov_dgf;
  array[2] vector[N_enzyme] priors_kcat;
  array[2] vector[N_km] priors_km;
  array[2] vector[N_competitive_inhibition] priors_ki;
  array[2] vector[N_allostery] priors_dissociation_constant;
  array[2] vector[N_allosteric_enzyme] priors_transfer_constant;
  array[2] vector[N_pme] priors_kcat_pme;
  // hardcoded priors (boundary conditions)
  array[2] vector[N_experiment_train] priors_psi_train;
  array[2, N_experiment_train] vector[N_pme] priors_conc_pme_train;
  array[2, N_experiment_train] vector[N_unbalanced] priors_conc_unbalanced_train;
  array[2, N_experiment_train] vector[N_enzyme] priors_conc_enzyme_train;
  array[2, N_experiment_train] vector[N_drain] priors_drain_train;
  // configuration
  array[N_experiment_train] vector<lower=0>[N_mic-N_unbalanced] conc_init;
  real rel_tol; 
  real abs_tol;
  real steady_state_threshold_abs;
  real steady_state_threshold_rel;
  int max_num_steps;
  int<lower=0,upper=1> likelihood;  // set to 0 for priors-only mode
  real drain_small_conc_corrector;
  real<lower=0> timepoint;
  array[N_quench_timepoints] real<lower=0> quench_timepoints;
  int<lower=0,upper=1> reject_non_steady;
}
transformed data {
  real initial_time = 0;
  matrix[N_metabolite, N_metabolite] prior_cov_dgf_chol = cholesky_decompose(prior_cov_dgf);
}
parameters {
  vector[N_metabolite] dgf;
  vector[N_enzyme] log_kcat_z;
  vector[N_km] log_km_z;
  vector[N_pme] log_kcat_pme_z;
  vector[N_competitive_inhibition] log_ki_z;
  vector[N_allostery] log_dissociation_constant_z;
  vector[N_allosteric_enzyme] log_transfer_constant_z;
  vector[N_experiment_train] psi_train_z;
  real alpha;
}
transformed parameters {
  // rescale
  vector[N_km] km = unz_log_1d(priors_km, log_km_z);
  vector[N_competitive_inhibition] ki = unz_log_1d(priors_ki, log_ki_z);
  vector[N_enzyme] kcat = unz_log_1d(priors_kcat, log_kcat_z);
  vector[N_allostery] dissociation_constant = unz_log_1d(priors_dissociation_constant, log_dissociation_constant_z);
  vector[N_allosteric_enzyme] transfer_constant = unz_log_1d(priors_transfer_constant, log_transfer_constant_z);
  vector[N_pme] kcat_pme = unz_log_1d(priors_kcat_pme, log_kcat_pme_z);
  vector[N_experiment_train] psi_train = unz_1d(priors_psi_train, psi_train_z);
    }
model {
  alpha ~ normal(0, 1);
}
generated quantities {
  // generate z-values for experiment specific parameters
  array[N_experiment_train] vector[N_drain] drain_train_z = rep_array(rep_vector(std_normal_rng(), N_drain), N_experiment_train);
  array[N_experiment_train] vector[N_enzyme] log_conc_enzyme_train_z = rep_array(rep_vector(std_normal_rng(), N_enzyme), N_experiment_train);  
  array[N_experiment_train] vector[N_pme] log_conc_pme_train_z = rep_array(rep_vector(std_normal_rng(), N_pme), N_experiment_train);  
  array[N_experiment_train] vector[N_unbalanced] log_conc_unbalanced_train_z =  rep_array(rep_vector(std_normal_rng(), N_unbalanced), N_experiment_train);
 // convert them using th prior to the correct values
  array[N_experiment_train] vector[N_drain] drain_train = unz_2d(priors_drain_train, drain_train_z);
  array[N_experiment_train] vector[N_enzyme] conc_enzyme_train = unz_log_2d(priors_conc_enzyme_train, log_conc_enzyme_train_z);
  array[N_experiment_train] vector[N_unbalanced] conc_unbalanced_train = unz_log_2d(priors_conc_unbalanced_train, log_conc_unbalanced_train_z);
  array[N_experiment_train] vector[N_unbalanced] conc_unbalanced_quench = rep_array(rep_vector(1e-5, N_unbalanced), N_experiment_train);
  array[N_experiment_train] vector[N_pme] conc_pme_train = unz_log_2d(priors_conc_pme_train, log_conc_pme_train_z);
  // define values for final states 
  array[N_experiment_train] vector[N_reaction] flux_train;
  array[N_experiment_train] vector<lower=0>[N_mic] conc_train;
  array[N_quench_timepoints, N_experiment_train] vector<lower=0>[N_mic] conc_quench;
  // transform
  array[N_experiment_train] vector[N_edge] dgr_train;

  for (e in 1:N_experiment_train){
    dgr_train[e] = get_dgr(S, dgf, temperature_train[e], mic_to_met, water_stoichiometry, transported_charge, psi_train[e]);
    flux_train[e] = rep_vector(0, N_reaction);
    vector[N_enzyme] conc_enzyme_experiment = conc_enzyme_train[e];
    vector[N_pme] conc_pme_experiment = conc_pme_train[e];
    array[1] vector[N_mic-N_unbalanced] conc_balanced_experiment;
    array[N_quench_timepoints] vector[N_mic-N_unbalanced] conc_quenched_experiment;
    int N_eko_experiment = measure_ragged(enzyme_knockout_train_bounds, e);
    int N_pko_experiment = measure_ragged(pme_knockout_train_bounds, e);
    if (N_eko_experiment > 0){
      array[N_eko_experiment] int eko_experiment =
        extract_ragged(enzyme_knockout_train_long, enzyme_knockout_train_bounds, e);
      conc_enzyme_experiment[eko_experiment] = rep_vector(0, N_eko_experiment);
    }
    if (N_pko_experiment > 0){
      array[N_pko_experiment] int pko_experiment =
        extract_ragged(pme_knockout_train_long, pme_knockout_train_bounds, e);
      conc_pme_experiment[pko_experiment] = rep_vector(0, N_pko_experiment);
    }
    conc_balanced_experiment =
      ode_bdf_tol(dbalanced_dt,
                  conc_init[e],
                  initial_time,
                  {timepoint},
                  rel_tol, 
                  abs_tol,
                  max_num_steps,
                  conc_unbalanced_train[e],
                  balanced_mic_ix,
                  unbalanced_mic_ix,
                  conc_enzyme_experiment,
                  dgr_train[e],
                  kcat,
                  km,
                  ki,
                  transfer_constant,
                  dissociation_constant,
                  kcat_pme,
                  conc_pme_experiment,
                  drain_train[e],
                  temperature_train[e],
                  drain_small_conc_corrector,
                  S,
                  subunits,
                  edge_type,
                  edge_to_enzyme,
                  edge_to_drain,
                  ci_mic_ix,
                  sub_km_ix_by_edge_long,
                  sub_km_ix_by_edge_bounds,
                  prod_km_ix_by_edge_long,
                  prod_km_ix_by_edge_bounds,
                  sub_by_edge_long,
                  sub_by_edge_bounds,
                  prod_by_edge_long,
                  prod_by_edge_bounds,
                  ci_ix_long,
                  ci_ix_bounds,
                  allostery_ix_long,
                  allostery_ix_bounds,
                  allostery_type,
                  allostery_mic,
                  edge_to_tc,
                  phosphorylation_ix_long,
                  phosphorylation_ix_bounds,
                  phosphorylation_type,
                  phosphorylation_pme);
    conc_train[e, balanced_mic_ix] = conc_balanced_experiment[1];
    conc_train[e, unbalanced_mic_ix] = conc_unbalanced_train[e];
        vector[N_edge] edge_flux = get_edge_flux(conc_train[e],
                                             conc_enzyme_experiment,
                                             dgr_train[e],
                                             kcat,
                                             km,
                                             ki,
                                             transfer_constant,
                                             dissociation_constant,
                                             kcat_pme,
                                             conc_pme_experiment,
                                             drain_train[e],
                                             temperature_train[e],
                                             drain_small_conc_corrector,
                                             S,
                                             subunits,
                                             edge_type,
                                             edge_to_enzyme,
                                             edge_to_drain,
                                             ci_mic_ix,
                                             sub_km_ix_by_edge_long,
                                             sub_km_ix_by_edge_bounds,
                                             prod_km_ix_by_edge_long,
                                             prod_km_ix_by_edge_bounds,
                                             sub_by_edge_long,
                                             sub_by_edge_bounds,
                                             prod_by_edge_long,
                                             prod_by_edge_bounds,
                                             ci_ix_long,
                                             ci_ix_bounds,
                                             allostery_ix_long,
                                             allostery_ix_bounds,
                                             allostery_type,
                                             allostery_mic,
                                             edge_to_tc,
                                             phosphorylation_ix_long,
                                             phosphorylation_ix_bounds,
                                             phosphorylation_type,
                                             phosphorylation_pme);
    for (j in 1:N_edge)
      flux_train[e, edge_to_reaction[j]] += edge_flux[j];  
    conc_quenched_experiment =
        ode_bdf_tol(dbalanced_dt,
                  conc_balanced_experiment[1],
                  initial_time,
                  quench_timepoints,
                  rel_tol, 
                  abs_tol,
                  max_num_steps,
                  conc_unbalanced_quench[e],
                  balanced_mic_ix,
                  unbalanced_mic_ix,
                  conc_enzyme_experiment,
                  dgr_train[e],
                  kcat,
                  km,
                  ki,
                  transfer_constant,
                  dissociation_constant,
                  kcat_pme,
                  conc_pme_experiment,
                  drain_train[e],
                  temperature_train[e],
                  drain_small_conc_corrector,
                  S,
                  subunits,
                  edge_type,
                  edge_to_enzyme,
                  edge_to_drain,
                  ci_mic_ix,
                  sub_km_ix_by_edge_long,
                  sub_km_ix_by_edge_bounds,
                  prod_km_ix_by_edge_long,
                  prod_km_ix_by_edge_bounds,
                  sub_by_edge_long,
                  sub_by_edge_bounds,
                  prod_by_edge_long,
                  prod_by_edge_bounds,
                  ci_ix_long,
                  ci_ix_bounds,
                  allostery_ix_long,
                  allostery_ix_bounds,
                  allostery_type,
                  allostery_mic,
                  edge_to_tc,
                  phosphorylation_ix_long,
                  phosphorylation_ix_bounds,
                  phosphorylation_type,
                  phosphorylation_pme);
    for (t in 1:N_quench_timepoints){
        conc_quench[t, e, balanced_mic_ix] = conc_quenched_experiment[t];
        conc_quench[t, e, unbalanced_mic_ix] = conc_unbalanced_quench[e];
        }
     }
}