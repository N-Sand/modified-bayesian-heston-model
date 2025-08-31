// Inputs are log returns; we model latent log variance (log_v)
data {
    int<lower=1> T;              // Number of observations
    vector[T] returns;           // Log returns (observed)
    int<lower=0> T_future;       // Steps to simulate forward
}

parameters {

    real mu;                     // Drift term
    real<lower=0> kappa;         // Mean reversion speed
    real log_theta;              // Long-run variance
    real<lower=0> sigma;         // Vol of vol
    real rho_raw;                // Return-volatility correlation

    // thresholding volaility jumps

    real log_v1;                 // Initial log variance
    vector[T - 1] log_v_std;     // Standardized latent log variance innovations
    vector[T] eps_v;             // Innovations for volatility
    vector[T] eps_r;             // Innovations for returns

}

transformed parameters {

    // volatility
    vector[T] log_v;
    vector[T] v;
    vector[T] sigma_t;
    vector[T] r_mu;
    vector[T] r_sd;

    real rho = tanh(rho_raw);    // Transform raw correlation to [-1, 1]

    log_v[1] = log_v1;
    for (t in 2:T) {
        real drift = kappa * (log_theta - log_v[t - 1]);
        log_v[t] = log_v[t - 1] + drift + sigma * log_v_std[t - 1];
    }

    v = exp(log_v);
    sigma_t = sqrt(v);
    for (t in 1:T) {
        r_mu[t] = mu - 0.5 * v[t];
    }
    r_sd = sigma_t;
}

model {
    // Priors
    mu ~ normal(0, 0.001);
    kappa ~ normal(1, 0.1);
    log_theta ~ normal(-9.5, 1);
    sigma ~ normal(0.2, 0.1);
    rho_raw ~ normal(-0.8, 0.2);

    eps_v ~ std_normal();
    eps_r ~ std_normal();

    log_v1 ~ normal(log_theta, 0.1);
    log_v_std ~ std_normal();

    // Return likelihood with correlated shocks
    for (t in 1:T) {
        real innovation = rho * eps_v[t] + sqrt(1 - rho^2) * eps_r[t];
        returns[t] ~ normal(r_mu[t], sigma_t[t]);
        target += -0.5 * square(innovation);  // log density of std normal
    }


}

generated quantities {
    vector[T_future] returns_future;
    vector[T_future] log_v_future;
    vector[T_future] v_future;


    real last_log_v = log_v[T];
    for (t in 1:T_future) {
        real v_prev = exp(last_log_v);
        real next_eps_v = normal_rng(0, 1);
        real next_eps_r = normal_rng(0, 1);
        real next_log_v = last_log_v + kappa * (log_theta - last_log_v) + sigma * next_eps_v;
        log_v_future[t] = next_log_v;
        v_future[t] = exp(next_log_v);

        returns_future[t] = normal_rng(
                                mu - 0.5 * v_future[t],
                                sqrt(v_future[t])
                                );

        last_log_v = next_log_v;
    }
}
