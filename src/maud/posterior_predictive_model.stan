functions{
#include functions.stan
  vector get_flux(vector conc,
                  vector enz,
                  vector km,
                  vector drain,
                  int[,] km_lookup,
                  matrix S,
                  int[] edge_type,
                  int[] edge_to_drain,
                  int[] edge_to_enzyme,
                  vector kcat,
                  vector keq,
                  int[] ix_ci,
                  int[] ix_ai,
                  int[] ix_aa,
                  int[] ix_pa,
                  int[] ix_pi,
                  int[] n_ci,
                  int[] n_ai,
                  int[] n_aa,
                  int[] n_pa,
                  int[] n_pi,
                  vector ki,
                  vector diss_t,
                  vector diss_r,
                  vector transfer_constant,
                  int[] subunits,
                  vector phos_kcat,
                  vector phos_conc){
    vector[cols(S)] out;
    int pos_ci = 1;
    int pos_ai = 1;
    int pos_aa = 1;
    int pos_tc = 1;
    int pos_pa = 1;
    int pos_pi = 1;
    for (j in 1:cols(S)){
      int n_mic_j = get_n_mic_for_edge(S, j, edge_type[j]);
      int mics_j[n_mic_j] = get_mics_for_edge(S, j, edge_type[j]);
      if (edge_type[j] == 1){  // reversible enzyme...
        vector[n_mic_j] km_j = km[km_lookup[mics_j, j]];
        real kcat_j = kcat[edge_to_enzyme[j]];
        real free_enzyme_ratio_denom = get_Dr_common_rate_law(conc[mics_j], km_j, S[mics_j, j]);
        if (n_ci[j] > 0){  // competitive inhibition
          int comp_inhs_j[n_ci[j]] = segment(ix_ci, pos_ci, n_ci[j]);
          vector[n_ci[j]] ki_j = segment(ki, pos_ci, n_ci[j]);
          free_enzyme_ratio_denom += sum(conc[comp_inhs_j] ./ ki_j);
          pos_ci += n_ci[j];
        }
        real free_enzyme_ratio = inv(free_enzyme_ratio_denom);
        out[j] = enz[edge_to_enzyme[j]] * free_enzyme_ratio * get_Tr(conc[mics_j], km_j, S[mics_j, j], kcat_j, keq[j]);
        if ((n_ai[j] > 0) || (n_aa[j] > 0)){  // allosteric regulation
          real Q_num = 1;
          real Q_denom = 1;
          if (n_ai[j] > 0){
            int allo_inhs_j[n_ai[j]] = segment(ix_ai, pos_ai, n_ai[j]);
            vector[n_ai[j]] diss_t_j = segment(diss_t, pos_ai, n_ai[j]);
            Q_num += sum(conc[allo_inhs_j] ./ diss_t_j);
            pos_ai += n_ai[j];
          }
          if (n_aa[j] > 0){
            int allo_acts_j[n_aa[j]] = segment(ix_aa, pos_aa, n_aa[j]);
            vector[n_aa[j]] diss_r_j = segment(diss_r, pos_aa, n_aa[j]);
            Q_denom += sum(conc[allo_acts_j] ./ diss_r_j);
            pos_aa += n_aa[j];
          }
          out[j] *= inv(1 + transfer_constant[pos_tc] * (free_enzyme_ratio * Q_num / Q_denom) ^ subunits[j]);
          pos_tc += 1;
        }
        if ((n_pi[j] > 0) || (n_pa[j] > 0)){  // phosphorylation
          real alpha = 0;
          real beta = 0;
          if (n_pa[j] > 0){
            int phos_acts_j[n_pa[j]] = segment(ix_pa, pos_pa, n_pa[j]);
            beta = sum(phos_kcat[phos_acts_j] .* phos_conc[phos_acts_j]);
            pos_pa += n_pa[j];
          }
          if (n_pi[j] > 0){
            int phos_inhs_j[n_pi[j]] = segment(ix_pi, pos_pi, n_pi[j]);
            alpha = sum(phos_kcat[phos_inhs_j] .* phos_conc[phos_inhs_j]);
            pos_pi += n_pi[j];
          }
          out[j] *= 1 / (1 + (alpha / beta) ^ subunits[j]);  // TODO: what if beta is zero and alpha is non-zero?
        }
      }
      else if (edge_type[j] == 2){  // drain...
        out[j] = drain[edge_to_drain[j]] * prod(conc[mics_j] ./ (conc[mics_j] + 1e-6));
      }
      else if (edge_type[j] == 3){  // irreversible modular rate law...
        vector[n_mic_j] km_j = km[km_lookup[mics_j, j]];
        real kcat_j = kcat[edge_to_enzyme[j]];
        real free_enzyme_ratio_denom = get_Dr_common_rate_law_irreversible(conc[mics_j], km_j, S[mics_j, j]);
        if (n_ci[j] > 0){  // competitive inhibition
          int comp_inhs_j[n_ci[j]] = segment(ix_ci, pos_ci, n_ci[j]);
          vector[n_ci[j]] ki_j = segment(ki, pos_ci, n_ci[j]);
          free_enzyme_ratio_denom += sum(conc[comp_inhs_j] ./ ki_j);
          pos_ci += n_ci[j];
        }
        real free_enzyme_ratio = inv(free_enzyme_ratio_denom);
        out[j] = enz[edge_to_enzyme[j]] * free_enzyme_ratio * get_Tr_irreversible(conc[mics_j], km_j, S[mics_j, j], kcat_j);
        if ((n_ai[j] > 0) || (n_aa[j] > 0)){  // allosteric regulation
          real Q_num = 1;
          real Q_denom = 1;
          if (n_ai[j] > 0){
            int allo_inhs_j[n_ai[j]] = segment(ix_ai, pos_ai, n_ai[j]);
            vector[n_ai[j]] diss_t_j = segment(diss_t, pos_ai, n_ai[j]);
            Q_num += sum(conc[allo_inhs_j] ./ diss_t_j);
            pos_ai += n_ai[j];
          }
          if (n_aa[j] > 0){
            int allo_acts_j[n_aa[j]] = segment(ix_aa, pos_aa, n_aa[j]);
            vector[n_aa[j]] diss_r_j = segment(diss_r, pos_aa, n_aa[j]);
            Q_denom += sum(conc[allo_acts_j] ./ diss_r_j);
            pos_aa += n_aa[j];
          }
          out[j] *= inv(1 + transfer_constant[pos_tc] * (free_enzyme_ratio * Q_num / Q_denom) ^ subunits[j]);
          pos_tc += 1;
        }
        if ((n_pi[j] > 0) || (n_pa[j] > 0)){  // phosphorylation
          real alpha = 0;
          real beta = 0;
          if (n_pa[j] > 0){
            int phos_acts_j[n_pa[j]] = segment(ix_pa, pos_pa, n_pa[j]);
            beta = sum(phos_kcat[phos_acts_j] .* phos_conc[phos_acts_j]);
            pos_pa += n_pa[j];
          }
          if (n_pi[j] > 0){
            int phos_inhs_j[n_pi[j]] = segment(ix_pi, pos_pi, n_pi[j]);
            alpha = sum(phos_kcat[phos_inhs_j] .* phos_conc[phos_inhs_j]);
            pos_pi += n_pi[j];
          }
          out[j] *= 1 / (1 + (alpha / beta) ^ subunits[j]);  // TODO: what if beta is zero and alpha is non-zero?
        }
      }
      else reject("Unknown edge type ", edge_type[j]);
    }
    return out;
  }
  vector dbalanced_dt(real time,
                      vector current_balanced,
                      vector unbalanced,
                      int[] balanced_ix,
                      int[] unbalanced_ix,
                      vector enz,
                      vector km,
                      vector drain,
                      int[,] km_lookup,
                      matrix S,
                      int[] edge_type,
                      int[] edge_to_drain,
                      int[] edge_to_enzyme,
                      vector kcat,
                      vector keq,
                      int[] ix_ci,
                      int[] ix_ai,
                      int[] ix_aa,
                      int[] ix_pa,
                      int[] ix_pi,
                      int[] n_ci,
                      int[] n_ai,
                      int[] n_aa,
                      int[] n_pa,
                      int[] n_pi,
                      vector ki,
                      vector diss_t,
                      vector diss_r,
                      vector transfer_constant,
                      int[] subunits,
                      vector kcat_phos,
                      vector conc_phos){
    vector[rows(current_balanced)+rows(unbalanced)] current_concentration;
    current_concentration[balanced_ix] = current_balanced;
    current_concentration[unbalanced_ix] = unbalanced;
    vector[rows(S)] flux = get_flux(current_concentration,
                                    enz, km, drain, km_lookup, S, edge_type, edge_to_drain, edge_to_enzyme, kcat, keq,
                                    ix_ci, ix_ai, ix_aa, ix_pa, ix_pi, n_ci, n_ai, n_aa, n_pa, n_pi,
                                    ki, diss_t, diss_r, transfer_constant, subunits, kcat_phos, conc_phos);
    return (S * flux)[balanced_ix];
  }
}
data {
  // dimensions
  int<lower=1> N_mic;
  int<lower=1> N_unbalanced;
  int<lower=1> N_metabolite;
  int<lower=1> N_km;
  int<lower=1> N_reaction;
  int<lower=1> N_enzyme;
  int<lower=0> N_drain;
  int<lower=1> N_edge;
  int<lower=0> N_phosphorylation_enzymes;
  int<lower=1> N_experiment;
  int<lower=0> N_ki;
  int<lower=0> N_ai;
  int<lower=0> N_aa;
  int<lower=0> N_ae;
  int<lower=0> N_pa;
  int<lower=0> N_pi;
  // hardcoded priors
  array[2] vector[N_metabolite] priors_dgf;
  array[2] vector[N_enzyme] priors_kcat;
  array[2] vector[N_km] priors_km;
  array[2] vector[N_ki] priors_ki;
  array[2] vector[N_ai] priors_diss_t;
  array[2] vector[N_aa] priors_diss_r;
  array[2] vector[N_ae] priors_transfer_constant;
  array[2] vector[N_phosphorylation_enzymes] priors_kcat_phos;
  array[2, N_experiment] vector[N_phosphorylation_enzymes] priors_conc_phos;
  array[2, N_experiment] vector[N_unbalanced] priors_conc_unbalanced;
  array[2, N_experiment] vector[N_enzyme] priors_conc_enzyme;
  array[2, N_experiment] vector[N_drain] priors_drain;
  // network properties
  int<lower=1,upper=N_mic> unbalanced_mic_ix[N_unbalanced];
  int<lower=1,upper=N_mic> balanced_mic_ix[N_mic-N_unbalanced];
  matrix[N_mic, N_edge] S;
  int<lower=1,upper=3> edge_type[N_edge];  // 1 = reversible modular rate law, 2 = drain
  int<lower=0,upper=N_enzyme> edge_to_enzyme[N_edge];  // 0 if drain
  int<lower=0,upper=N_drain> edge_to_drain[N_edge];  // 0 if enzyme
  int<lower=0,upper=N_reaction> edge_to_reaction[N_edge];
  int<lower=1,upper=N_metabolite> mic_to_met[N_mic];
  vector[N_edge] water_stoichiometry;
  matrix<lower=0,upper=1>[N_experiment, N_enzyme] is_knockout;
  matrix<lower=0,upper=1>[N_experiment, N_phosphorylation_enzymes] is_phos_knockout;
  int<lower=0,upper=N_km> km_lookup[N_mic, N_edge];
  int<lower=0,upper=N_mic> n_ci[N_edge];
  int<lower=0,upper=N_mic> n_ai[N_edge];
  int<lower=0,upper=N_mic> n_aa[N_edge];
  int<lower=0,upper=N_phosphorylation_enzymes> n_pa[N_edge];
  int<lower=0,upper=N_phosphorylation_enzymes> n_pi[N_edge];
  int<lower=0,upper=N_mic> ix_ci[N_ki];
  int<lower=0,upper=N_mic> ix_ai[N_ai];
  int<lower=0,upper=N_mic> ix_aa[N_aa];
  int<lower=0,upper=N_phosphorylation_enzymes> ix_pa[N_pa];
  int<lower=0,upper=N_phosphorylation_enzymes> ix_pi[N_pi];
  int<lower=1> subunits[N_enzyme];
  // configuration
  vector<lower=0>[N_mic] conc_init[N_experiment];
  real rel_tol_forward; 
  vector[N_mic - N_unbalanced] abs_tol_forward;
  real rel_tol_backward; 
  vector[N_mic - N_unbalanced] abs_tol_backward; 
  real rel_tol_quadrature;
  real abs_tol_quadrature;
  int max_num_steps;
  int num_steps_between_checkpoints;
  int interpolation_polynomial;
  int solver_forward;
  int solver_backward;
  int<lower=0,upper=1> LIKELIHOOD;  // set to 0 for priors-only mode
  real<lower=0> timepoint;
}
transformed data {
  real initial_time = 0;
  matrix[N_experiment, N_enzyme] knockout = rep_matrix(1, N_experiment, N_enzyme) - is_knockout;
  matrix[N_experiment, N_phosphorylation_enzymes] phos_knockout =
    rep_matrix(1, N_experiment, N_phosphorylation_enzymes) - is_phos_knockout;
}
parameters {
  vector[N_km] km;
  vector[N_ki] ki;
  vector[N_enzyme] kcat;
  vector[N_ai] diss_t;
  vector[N_aa] diss_r;
  vector[N_ae] transfer_constant;
  vector[N_phosphorylation_enzymes] kcat_phos;
  vector[N_edge] keq;
}

generated quantities {
  array[N_experiment] vector<lower=0>[N_mic] conc;
  array[N_experiment] vector[N_reaction] flux;
  array[N_experiment] vector[N_phosphorylation_enzymes] conc_phos;
  array[N_experiment] vector[N_unbalanced] conc_unbalanced;
  array[N_experiment] vector[N_enzyme] conc_enzyme;
  array[N_experiment] vector[N_drain] drain;
  // Sampling experiment boundary conditions from priors
  for (e in 1:N_experiment){
    drain[e] = to_vector(normal_rng(priors_drain[1,e], priors_drain[2,e]));
    conc_phos[e] = to_vector(lognormal_rng(log(priors_conc_phos[1,e]), priors_conc_phos[2,e]));
    conc_unbalanced[e] = to_vector(lognormal_rng(log(priors_conc_unbalanced[1,e]), priors_conc_unbalanced[2,e]));
    conc_enzyme[e] = to_vector(lognormal_rng(log(priors_conc_enzyme[1,e]), priors_conc_enzyme[2,e]));
  }
  // Simulation of experiments
  for (e in 1:N_experiment){
    flux[e] = rep_vector(0, N_reaction);
    real timepoints[2] = {timepoint, timepoint + 10};
    vector[N_enzyme] conc_enzyme_experiment = conc_enzyme[e] .* knockout[e]';
    vector[N_phosphorylation_enzymes] conc_phos_experiment = conc_phos[e] .* phos_knockout[e]';
    vector[N_mic-N_unbalanced] conc_balanced[2] =
      ode_adjoint_tol_ctl(dbalanced_dt,
                  conc_init[e, balanced_mic_ix],
                  initial_time,
                  timepoints,
                  rel_tol_forward, 
                  abs_tol_forward,
                  rel_tol_backward, 
                  abs_tol_backward, 
                  rel_tol_quadrature,
                  abs_tol_quadrature,
                  max_num_steps,
                  num_steps_between_checkpoints,
                  interpolation_polynomial,
                  solver_forward,
                  solver_backward,
                  conc_unbalanced[e,:],
                  balanced_mic_ix,
                  unbalanced_mic_ix,
                  conc_enzyme_experiment,
                  km,
                  drain[e,:],
                  km_lookup,
                  S,
                  edge_type,
                  edge_to_drain,
                  edge_to_enzyme,
                  kcat,
                  keq,
                  ix_ci,
                  ix_ai,
                  ix_aa,
                  ix_pa,
                  ix_pi,
                  n_ci,
                  n_ai,
                  n_aa,
                  n_pa,
                  n_pi,
                  ki,
                  diss_t,
                  diss_r,
                  transfer_constant,
                  subunits,
                  kcat_phos,
                  conc_phos_experiment);
    conc[e, balanced_mic_ix] = conc_balanced[1];
    conc[e, unbalanced_mic_ix] = conc_unbalanced[e,:];
    {
    vector[N_edge] flux_edge = get_flux(conc[e],
                                        conc_enzyme_experiment,
                                        km,
                                        drain[e,:],
                                        km_lookup,
                                        S,
                                        edge_type,
                                        edge_to_drain,
                                        edge_to_enzyme,
                                        kcat,
                                        keq,
                                        ix_ci,
                                        ix_ai,
                                        ix_aa,
                                        ix_pa,
                                        ix_pi,
                                        n_ci,
                                        n_ai,
                                        n_aa,
                                        n_pa,
                                        n_pi,
                                        ki,
                                        diss_t,
                                        diss_r,
                                        transfer_constant,
                                        subunits,
                                        kcat_phos,
                                        conc_phos_experiment);
    for (j in 1:N_edge)
      flux[e, edge_to_reaction[j]] += flux_edge[j];
    }
  }
}
