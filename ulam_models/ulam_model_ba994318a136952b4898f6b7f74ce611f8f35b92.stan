data {
  int<lower=0> N;
  vector[N] x;
  vector[N] y;
}

parameters {
  real intercept;
  real slope;
  real<lower=0.01> error;
}

model {
  
  for (i in 1:N) {
    y[i] ~ normal(x[i] * slope + intercept, error);
  }
}

generated quantities {
  vector[N] log_lik;
  vector[N] y__hat;
  
  for (i in 1:N) {
    log_lik[i] = normal_lcdf(y[i] | x[i] * slope + intercept, error);
    y__hat[i] = normal_rng(x[i] * slope + intercept, error);
  }
}