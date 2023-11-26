data {
  int<lower=0> N;
  // ------------------------------------------------------------
  // Variable x modified to hold missing data by Ulam.
  //  START of code generated for missing data
  // ------------------------------------------------------------
  int<lower=0, upper=N> N__x__missing;
  int<lower=0, upper=N> N__x__not_missing;
  array[N] int<lower=0, upper=1> x__is_missing;
  array[N] int<lower=0, upper=N> x__missing_data_index;
  vector[N__x__not_missing] x__not_missing;
  // ------------------------------------------------------------
  //  END of code generated for missing data
  // ------------------------------------------------------------

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

}

parameters {
  // ------------------------------------------------------------
  // Variable x modified to hold missing data by Ulam.
  //  START of code generated for missing data
  // ------------------------------------------------------------
  vector[N__x__missing] x__missing;
  // ------------------------------------------------------------
  //  END of code generated for missing data
  // ------------------------------------------------------------

  // ------------------------------------------------------------
  // Variable y modified to hold missing data by Ulam.
  //  START of code generated for missing data
  // ------------------------------------------------------------
  vector[N__y__missing] y__missing;
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
    if (x__is_missing[i]) {
      x__missing[x__missing_data_index[i]] ~ normal(mu_x, sigma_x);
    } else {
      x__not_missing[x__missing_data_index[i]] ~ normal(mu_x, sigma_x);
    }
  }
  for (i in 1:N) {
    if (x__is_missing[i]) {
      if (y__is_missing[i]) {
        y__missing[y__missing_data_index[i]] ~ normal(x__missing[x__missing_data_index[i]] * slope + intercept, error);
      } else {
        y__not_missing[y__missing_data_index[i]] ~ normal(x__missing[x__missing_data_index[i]] * slope + intercept, error);
      }
    } else {
      if (y__is_missing[i]) {
        y__missing[y__missing_data_index[i]] ~ normal(x__not_missing[x__missing_data_index[i]] * slope + intercept, error);
      } else {
        y__not_missing[y__missing_data_index[i]] ~ normal(x__not_missing[x__missing_data_index[i]] * slope + intercept, error);
      }
    }
  }
}