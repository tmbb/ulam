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

generated quantities {
  vector[N] log_lik;
  vector[N] y__hat;
}