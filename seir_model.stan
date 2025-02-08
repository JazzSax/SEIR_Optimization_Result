functions {
  array[] real seir(real t, array[] real y, array[] real theta, 
                    array[] real x_r, array[] int x_i) {
    real N = x_i[1];
    real lambda = theta[1];
    real mu = theta[2];
    real sigma = theta[3];
    real i0 = theta[4];
    real e0 = theta[5];
    
    real S = y[1];
    real E = y[2];
    real I = y[3];
    real R = y[4];
    
    real dS_dt = -lambda * I * S / N;
    real dE_dt = lambda * I * S / N - sigma * E;
    real dI_dt = sigma * E - mu * I;
    real dR_dt = mu * I;
    
    return {dS_dt, dE_dt, dI_dt, dR_dt};
  }
}

data {
  int<lower=1> n_days;
  real t0;
  array[n_days] real ts;
  int<lower=1> N;
  array[n_days] int<lower=0> cases;
  real<lower=0> i0;   // Initial number of infectious individuals
  real<lower=0> e0;   // Initial number of exposed individuals
  real<lower=0> r0;   // Initial number of recovered individuals
}


transformed data {
  array[0] real x_r;
  array[1] int x_i = {N};
}

parameters {
  real<lower=0.1, upper=1> mu; 
  real<lower=0.1, upper=1> lambda;
  real<lower=0.1, upper=1> sigma;
  real<lower=1e-5> phi_inv;
  real<lower=0, upper=1> reporting_D;
}

transformed parameters {
  array[n_days, 4] real y;
  array[n_days - 1] real incidence;
  real phi = fmax(1e-5, 1 / phi_inv);
  array[5] real tt = {lambda, mu, sigma, i0, e0};
  
  // Use the user-defined initial conditions for exposed, infected, and recovered
  y = integrate_ode_rk45(seir, {N - i0 - e0 - r0, e0, i0, r0}, t0, ts, tt, x_r, x_i, 1e-6, 1e-6, 1e6);
  
  for (i in 1:n_days-1) {
    real new_incidence = -(y[i + 1, 2] - y[i, 2] + y[i + 1, 1] - y[i, 1]) * reporting_D;
    incidence[i] = fmax(1e-6, new_incidence);
  }
}

model {
  lambda ~ normal(0.5, 0.25); 
  mu ~ normal(0.25, 0.1); 
  sigma ~ normal(0.2, 0.1);
  phi_inv ~ exponential(5);
  e0 ~ cauchy(0, 20);
  reporting_D ~ beta(2, 5);
  
  cases[1:(n_days - 1)] ~ neg_binomial_2(incidence, phi);
}

generated quantities {
  real recovery_time = 1 / mu;
  real incubation_time = 1 / sigma;
  real infectious_rate = lambda;
  real incubation_rate = sigma;
  real recovery_rate = mu;
}
