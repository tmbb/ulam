data {
  int<lower=0> N;
  array[N] int<lower=0> years;
  array[N] int<lower=N> disasters;
}

parameters {
  real<lower=min(years), upper=max(years)> switchpoint;
  real<lower=0.0> early_rate;
  real<lower=0.0> late_rate;
}

model {
  for (i in 1:N) {
    disasters[i] ~ poisson(early_rate + late_rate);
  }
}