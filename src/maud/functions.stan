functions {
  vector unz_1d(vector[] mnsd, vector z){
    /* 
      Recover a real-valued vector from a 1xn vector z of z scores and a 2xn
      array mnsd of vectors consisting of the mean and standard deviation of
      each element. 
    */
    return mnsd[1] + mnsd[2] .* z;
  }

  vector unz_log_1d(vector[] mnsd, vector z){
    /* 
      Recover a positive-constrained vector from a 1xn vector z of z scores and
      a 2xn array mnsd of vectors consisting of the lognormal mean and standard
      deviation of each element. 
    */
    return exp(log(mnsd[1]) + mnsd[2] .* z);
  }

  vector[] unz_2d(vector[,] mnsd, vector[] z){
    /*
      Recover a mxn array of real-valued vectors from an mxn array z of vectors
      of z scores and a 2xmxn array mnsd of vectors consisting of mean and
      standard deviation arrays.
    */
    array[size(z)] vector [rows(z[1])] out;
    for (ex in 1:size(z)){
      out[ex] = unz_1d(mnsd[:,ex], z[ex]);
    }
    return out;
  }

  vector[] unz_log_2d(vector[,] mnsd, vector[] z){
    /*
      Recover a mxn array of positive-constrained vectors from an mxn array z of
      vectors of z scores and a 2xmxn array mnsd of vectors consisting of
      lognormal mean and standard deviation arrays.
    */
    array[size(z)] vector [rows(z[1])] out;
    for (ex in 1:size(z)){
      out[ex] = unz_log_1d(mnsd[:,ex], z[ex]);
    }
    return out;
  }

  vector get_keq(matrix S, vector dgf, int[] mic_to_met, vector water_stoichiometry){
    /*
        Calculate keqs from metabolite formation energies, assuming water's
        formation energy is known exactly.
    */
    real minus_RT = -0.008314 * 298.15;
    real dgf_water = -150.9;  // From http://equilibrator.weizmann.ac.il/metabolite?compoundId=C00001
    vector[cols(S)] delta_g = S' * dgf[mic_to_met] + water_stoichiometry * dgf_water;
    return exp(delta_g / minus_RT);
  }

  vector get_dgrs(matrix S, vector dgf, int[] mic_to_met, vector water_stoichiometry){
    /*
        Calculate dgr standard from metabolite formation energies, assuming water's
        formation energy is known exactly.
    */
    real minus_RT = -0.008314 * 298.15;
    real dgf_water = -150.9;  // From http://equilibrator.weizmann.ac.il/metabolite?compoundId=C00001
    vector[cols(S)] dgrs = S' * dgf[mic_to_met] + water_stoichiometry * dgf_water;
    return dgrs;
  }

  int check_steady_state(vector Sv, vector conc, real abs_thresh, real rel_thresh){
    /* Relative and absolute check for steady state. */
    vector[rows(conc)] rel_thresh_per_conc = conc * rel_thresh;
    int relative_check_failed = max(fabs(Sv) - rel_thresh_per_conc) > 0;
    int absolute_check_failed = max(fabs(Sv)) > abs_thresh;
    if (relative_check_failed) print("Sv ", Sv, " not within ", rel_thresh_per_conc, " of zero.");
    if (absolute_check_failed) print("Sv ", Sv, " not within ", abs_thresh, " of zero.");
    return (relative_check_failed || absolute_check_failed) ? 0 : 1;
  }

  int measure_ragged(int[,] bounds, int i){
    return bounds[i, 2] - bounds[i, 1] + 1;
  }

  int[] extract_ragged(int[] ix_long, int[,] bounds, int i){
  /*
    Extract the ith element of a ragged array stored in 1d array long.

    Make sure that members bounds[i, 1] to bounds[i, 2] of long form the
    required element!

   */
  return ix_long[bounds[i, 1]:bounds[i, 2]];
  }

  vector get_saturation(vector conc,
                        vector km,
                        vector free_enzyme_ratio,
                        array[N_edge_sub] int sub_km_ix_by_edge_long,
                        array[N_edge,2] int sub_km_ix_by_edge_bounds,
                        int[] sub_by_edge_long,
                        int[,] sub_by_edge_bounds,
                        int[] edge_type){
    int N_edge = size(sub_by_edge_bounds);
    vector[N_edge] prod_conc_over_km;
    for (f in 1:N_edge){
      if (edge_type[f] == 2){
        prod_conc_over_km[f] = 1;
        continue;
      }
      int N_sub = measure_ragged(sub_by_edge_bounds, f);
      array[N_sub] int sub_ix = extract_ragged(sub_by_edge_long, sub_by_edge_bounds, f);
      array[N_sub] int sub_km_ix = extract_ragged(sub_km_ix_by_edge_long, sub_km_ix_by_edge_bounds, f);
      prod_conc_over_km[f] = prod(conc[sub_ix] ./ km[sub_km_ix]);
    }
    return prod_conc_over_km .* free_enzyme_ratio;
  }

  vector get_free_enzyme_ratio(vector conc,
                               matrix S,
                               vector km,
                               vector ki,
                               int[] edge_type,
                               int[] sub_km_ix_by_edge_long,
                               int[,] sub_km_ix_by_edge_bounds,
                               int[] prod_km_ix_by_edge_long,
                               int[,] prod_km_ix_by_edge_bounds,
                               int[] sub_by_edge_long,
                               int[,] sub_by_edge_bounds,
                               int[] prod_by_edge_long,
                               int[,] prod_by_edge_bounds,
                               int[] ci_by_edge_long,
                               int[,] ci_by_edge_bounds){
    /* Find the proportion of enzyme that is free, for each edge. */
    int N_edge = cols(S);
    vector[N_edge] denom;
    for (f in 1:N_edge){
      if (edge_type[f] == 2){
        denom[f] = 1;
        continue;
      }
      int N_sub = measure_ragged(sub_by_edge_bounds, f);
      int N_prod = measure_ragged(prod_by_edge_bounds, f);
      int N_ci = measure_ragged(ci_by_edge_bounds, f);
      array[N_sub] int sub_ix = extract_ragged(sub_by_edge_long, sub_by_edge_bounds, f);
      array[N_sub] int sub_km_ix = extract_ragged(sub_km_ix_by_edge_long, sub_km_ix_by_edge_bounds, f);
      array[N_prod] int prod_ix = extract_ragged(prod_by_edge_long, prod_by_edge_bounds, f);
      vector[N_sub] sub_over_km = conc[sub_ix] ./ km[sub_km_ix];
      denom[f] = prod((rep_vector(1, N_sub) + sub_over_km) ^ fabs(S[sub_ix, f]));
      if (edge_type[f] == 1){
        array[N_prod] int prod_km_ix = extract_ragged(prod_km_ix_by_edge_long, prod_km_ix_by_edge_bounds, f);
        vector[N_prod] prod_over_km = conc[prod_ix] ./ km[prod_km_ix];
        denom[f] += prod((rep_vector(1, N_prod) + prod_over_km) ^ fabs(S[prod_ix, f])) - 1;
      }
      if (N_ci > 0){
        array[N_ci] int ci_ix = extract_ragged(ci_by_edge_long, ci_by_edge_bounds, f);
        denom[f] += sum(conc[ci_ix] ./ ki[ci_ix]);
      }
    }
    return inv(denom);
  }

  vector get_reversibility(vector dgr, matrix S, vector conc, int[] edge_type){
    real RT = 0.008314 * 298.15;
    int N_edge = cols(S);
    vector[N_edge] reaction_quotient = S' * log(conc);
    vector[N_edge] out;
    for (f in 1:N_edge){
      if (edge_type[f] == 1)
        out[f] = 1 - exp((dgr[f] + RT * reaction_quotient[f])/RT);
      else
        out[f] = 1;
    }
    return out;
  }

  vector get_allostery(vector conc,                               // one per mic
                       vector free_enzyme_ratio,                  // one per edge
                       vector tc,                                 // one per allosteric enzyme
                       vector dc,                                 // one per allostery
                       vector subunits,                           // one per edge
                       array[] int allostery_ix_long,             // - long and bounds encode a ragged
                       array[,] int allostery_ix_bounds,          //   array with one entry per edge
                       array[] int allostery_type,                // one per allostery
                       array[] int allostery_mic,                 // one per allostery
                       array[] int edge_to_tc                     // one per edge
  ){
    int N_edge = size(allostery_ix_bounds);
    vector[N_edge] out = rep_vector(1, N_edge);
    for (f in 1:N_edge){
      int N_allostery = measure_ragged(allostery_ix_bounds, f);
      if (N_allostery == 0){
        continue;
      }
      real Q_num = 1;
      real Q_denom = 1;
      real tc_edge = tc[edge_to_tc[f]];
      for (allostery in extract_ragged(allostery_ix_long, allostery_ix_bounds, f)){
        real conc_over_dc = conc[allostery_mic[allostery]] / dc[allostery];
        if (allostery_type == 1){ // inhibition
          Q_num += conc_over_dc;
        }
        else {  // activation
          Q_denom += conc_over_dc;
        }
      }
      out[f] = inv(1 + tc_edge * (free_enzyme_ratio[f] * (1+Q_num) / (1+Q_denom)) ^ subunits[f]);
    }
    return out;
  }

  vector get_phosphorylation(vector kcat_phos,
                             vector conc_phos,
                             array[] int phos_ix_long,
                             array[,] int phos_ix_bounds,
                             array[] int phos_type,
                             vector subunits){
    int N_edge = size(pa_ix_bounds);
    vector[N_edge] out = rep_vector(1, N_edge);
    for (f in 1:N_edge){
      N_phos = measure_ragged(phos_ix_bounds, f);
      if (N_phos == 0){
        continue;
      }
      real alpha = 0;
      real beta = 0;
      for (phos in extract_ragged(phos_ix_long, phos_ix_bounds, f)){
        real kcat_times_conc = kcat_phos[phos] * conc_phos[phos];
        if (phos_type[phos] == 1){ // inhibition
          alpha += kcat_times_conc;
        }
        else{
          beta += kcat_times_conc;
        }
        out[f] = inv(1 + (alpha / beta) ^ subunits[f]);
      }
    }
    return out;
  }

  vector get_drain_by_edge(vector drain,
                           vector conc,
                           int[] edge_to_drain,
                           int[] sub_by_edge_long,
                           int[,] sub_by_edge_bounds,
                           int[] edge_type){
    int N_edge = size(edge_type);
    vector[N_edge] out = rep_vector(1, N_edge);
    for (f in 1:N_edge){
      if (edge_type[f] == 2){
        int N_sub = measure_ragged(sub_by_edge_bounds, f);
        int subs[N_sub] = extract_ragged(sub_by_edge_long, sub_by_edge_bounds, f);
        out[f] = drain[edge_to_drain[f]] * prod(conc[subs] ./ (conc[subs] + 1e-6));
      }
    }
    return out;
  }

  vector get_vmax_by_edge(vector enzyme, vector kcat, int[] edge_to_enzyme, int[] edge_type){
    int N_edge = size(edge_to_enzyme);
    vector[N_edge] out = rep_vector(1, N_edge);
    for (f in 1:N_edge){
      if (edge_type[f] != 2){
        out[f] = enzyme[edge_to_enzyme[f]] * kcat[edge_to_enzyme[f]];
      }
    }
    return out;
  }

  vector get_edge_flux(vector conc,
                       vector enzyme,
                       vector dgr,
                       vector kcat,
                       vector km,
                       vector ki,
                       vector tc,
                       vector dt,
                       vector dr,
                       vector kcat_phos,
                       vector conc_phos,
                       vector drain,
                       matrix S,
                       vector subunits,
                       array[] int edge_type,
                       array[] int edge_to_enzyme,
                       array[] int edge_to_tc,
                       array[] int edge_to_drain,
                       array[] int sub_km_ix_by_edge_long,
                       array[,] int sub_km_ix_by_edge_bounds,
                       array[] int prod_km_ix_by_edge_long,
                       array[,] int prod_km_ix_by_edge_bounds,
                       array[] int sub_by_edge_long,
                       array[,] int sub_by_edge_bounds,
                       array[] int prod_by_edge_long,
                       array[,] int prod_by_edge_bounds,
                       array[] int ci_by_edge_long,
                       array[,] int ci_by_edge_bounds,
                       array[] int ai_ix_long,
                       array[,] int ai_ix_bounds,
                       array[] int aa_ix_long,
                       array[,] int aa_ix_bounds,
                       array[] int pa_ix_long,
                       array[,] int pa_ix_bounds,
                       array[] int pi_ix_long,
                       array[,] int pi_ix_bounds){
    int N_edge = cols(S);
    vector[N_edge] vmax = get_vmax_by_edge(enzyme, kcat, edge_to_enzyme, edge_type);
    vector[N_edge] reversibility = get_reversibility(dgr, S, conc, edge_type);
    vector[N_edge] free_enzyme_ratio = get_free_enzyme_ratio(conc,
                                                             S,
                                                             km,
                                                             ki,
                                                             edge_type,
                                                             sub_km_ix_by_edge_long,
                                                             sub_km_ix_by_edge_bounds,
                                                             prod_km_ix_by_edge_long,
                                                             prod_km_ix_by_edge_bounds,
                                                             sub_by_edge_long,
                                                             sub_by_edge_bounds,
                                                             prod_by_edge_long,
                                                             prod_by_edge_bounds,
                                                             ci_by_edge_long,
                                                             ci_by_edge_bounds);
    vector[N_edge] saturation = get_saturation(conc,
                                               km,
                                               free_enzyme_ratio,
                                               sub_km_ix_by_edge_long,
                                               sub_km_ix_by_edge_bounds,
                                               sub_by_edge_long,
                                               sub_by_edge_bounds,
                                               edge_type);
    vector[N_edge] allostery = get_allostery(conc,
                                             free_enzyme_ratio,
                                             tc,
                                             dt,
                                             dr,
                                             subunits,
                                             edge_to_tc,
                                             ai_ix_long,
                                             ai_ix_bounds,
                                             aa_ix_long,
                                             aa_ix_bounds);
    vector[N_edge] phosphorylation = get_phosphorylation(kcat_phos,
                                                         conc_phos,
                                                         pa_ix_long,
                                                         pa_ix_bounds,
                                                         pi_ix_long,
                                                         pi_ix_bounds,
                                                         subunits);
    vector[N_edge] drain_by_edge = get_drain_by_edge(drain,
                                                     conc,
                                                     edge_to_drain,
                                                     sub_by_edge_long,
                                                     sub_by_edge_bounds,
                                                     edge_type);
    return vmax .* saturation .* reversibility .* allostery .* phosphorylation .* drain_by_edge;
  }

  vector dbalanced_dt(real time,
                      vector current_balanced,
                      vector unbalanced,
                      int[] balanced_ix,
                      int[] unbalanced_ix,
                      vector enzyme,
                      vector dgr,
                      vector kcat,
                      vector km,
                      vector ki,
                      vector tc,
                      vector dt,
                      vector dr,
                      vector kcat_phos,
                      vector conc_phos,
                      vector drain,
                      matrix S,
                      vector subunits,
                      array[] int edge_type,
                      array[] int edge_to_enzyme,
                      array[] int edge_to_drain,
                      array[] int sub_km_ix_by_edge_long,
                      array[,] int sub_km_ix_by_edge_bounds,
                      array[] int prod_km_ix_by_edge_long,
                      array[,] int prod_km_ix_by_edge_bounds,
                      array[] int sub_by_edge_long,
                      array[,] int sub_by_edge_bounds,
                      array[] int prod_by_edge_long,
                      array[,] int prod_by_edge_bounds,
                      array[] int ci_ix_long,
                      array[,] int ci_ix_bounds,
                      array[] int allostery_ix_long,
                      array[,] int allostery_ix_bounds,
                      array[] int allostery_type,
                      array[] int allostery_mic,
                      array[] int edge_to_tc,
                      array[] int phosphorylation_ix_long,
                      array[,] int phosphorylation_ix_bounds,
                      array[] int phosphorylation_type

                      ){
    vector[rows(current_balanced)+rows(unbalanced)] current_concentration;
    current_concentration[balanced_ix] = current_balanced;
    current_concentration[unbalanced_ix] = unbalanced;
    vector[cols(S)] edge_flux = get_edge_flux(current_concentration,
                                              enzyme,
                                              dgr,
                                              kcat,
                                              km,
                                              ki,
                                              tc,
                                              dt,
                                              dr,
                                              kcat_phos,
                                              conc_phos,
                                              drain,
                                              S,
                                              subunits,
                                              edge_type,
                                              edge_to_enzyme,
                                              edge_to_tc,
                                              edge_to_drain,
                                              sub_km_ix_by_edge_long,
                                              sub_km_ix_by_edge_bounds,
                                              prod_km_ix_by_edge_long,
                                              prod_km_ix_by_edge_bounds,
                                              sub_by_edge_long,
                                              sub_by_edge_bounds,
                                              prod_by_edge_long,
                                              prod_by_edge_bounds,
                                              ci_by_edge_long,
                                              ci_by_edge_bounds,
                                              ai_ix_long,
                                              ai_ix_bounds,
                                              aa_ix_long,
                                              aa_ix_bounds,
                                              pa_ix_long,
                                              pa_ix_bounds,
                                              pi_ix_long,
                                              pi_ix_bounds);
    return (S * edge_flux)[balanced_ix];
  }
}
