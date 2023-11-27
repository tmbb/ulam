data {
  int<lower=0> N;
  vector[N] x;
  // ------------------------------------------------------------
  // Variable y modified to hold missing data by Ulam.
  //  START of code generated for missing data
  // ------------------------------------------------------------
  int<lower=0, upper=N> N__y__missing;
  int<lower=0, upper=N> N__y__not_missing;
  array[N] int<lower=0, upper=1> y__is_missing;
  array[N] int<lower=0, upper=N> y__missing_data_index;
  vector[N__y__not_missing] y__not_missing;
  // ------------------------------------------------------------
  //  END of code generated for missing data
  // ------------------------------------------------------------

  // ------------------------------------------------------------
  // Variable z modified to hold missing data by Ulam.
  //  START of code generated for missing data
  // ------------------------------------------------------------
  int<lower=0, upper=N> N__z__missing;
  int<lower=0, upper=N> N__z__not_missing;
  array[N] int<lower=0, upper=1> z__is_missing;
  array[N] int<lower=0, upper=N> z__missing_data_index;
  vector[N__z__not_missing] z__not_missing;
  // ------------------------------------------------------------
  //  END of code generated for missing data
  // ------------------------------------------------------------

}

parameters {
  // ------------------------------------------------------------
  // Variable y modified to hold missing data by Ulam.
  //  START of code generated for missing data
  // ------------------------------------------------------------
  vector[N__y__missing] y__missing;
  // ------------------------------------------------------------
  //  END of code generated for missing data
  // ------------------------------------------------------------

  // ------------------------------------------------------------
  // Variable z modified to hold missing data by Ulam.
  //  START of code generated for missing data
  // ------------------------------------------------------------
  vector[N__z__missing] z__missing;
  // ------------------------------------------------------------
  //  END of code generated for missing data
  // ------------------------------------------------------------

  real mu_x;
  real<lower=0.01> sigma_x;
  real intercept;
  real slope;
  real<lower=0.01> error;
}

model {
  for (i in 1:N) {
    x[i] ~ normal(mu_x, sigma_x);
  }
  for (i in 1:N) {
    if (y__is_missing[i]) {
      y__missing[y__missing_data_index[i]] ~ normal(x[i] * slope + intercept, error);
    } else {
      y__not_missing[y__missing_data_index[i]] ~ normal(x[i] * slope + intercept, error);
    }
  }
  for (i in 1:N) {
    if (z__is_missing[i]) {
      z__missing[z__missing_data_index[i]] ~ normal(x[i] * slope + intercept, error);
    } else {
      z__not_missing[z__missing_data_index[i]] ~ normal(x[i] * slope + intercept, error);
    }
  }
}

generated quantities {
  vector[N__y__not_missing] y__hat;
  vector[N__y__not_missing] y__log_lik;
  for (i in 1:N) {
    if (y__is_missing[i]) {
      
    } else {
      y__hat[y__missing_data_index[i]] = normal_rng(x[i] * slope + intercept, error);
      y__log_lik[y__missing_data_index[i]] = normal_lpdf(y__not_missing[y__missing_data_index[i]] | x[i] * slope + intercept, error);
    }
  }
  vector[N__z__not_missing] z__hat;
  vector[N__z__not_missing] z__log_lik;
  for (i in 1:N) {
    if (z__is_missing[i]) {
      
    } else {
      z__hat[z__missing_data_index[i]] = normal_rng(x[i] * slope + intercept, error);
      z__log_lik[z__missing_data_index[i]] = normal_lpdf(z__not_missing[z__missing_data_index[i]] | x[i] * slope + intercept, error);
    }
  }
}