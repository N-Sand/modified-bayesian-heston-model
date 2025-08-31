// Heston model in Stan using log returns and log volatility (log variance)
functions {
    // Fourier series without constant term
    // Takes separate arrays for sine and cosine coefficients
    real fourier_series(real x, vector sin_coeffs, vector cos_coeffs, real period) {
        real result = 0;
        int n_sin = num_elements(sin_coeffs);
        int n_cos = num_elements(cos_coeffs);
        
        // Add sine components
        for (i in 1:n_sin) {
        result += sin_coeffs[i] * sin(2 * pi() * i * x / period);
        }
        
        // Add cosine components
        for (i in 1:n_cos) {
        result += cos_coeffs[i] * cos(2 * pi() * i * x / period);
        }
        
        return result;
    }
    
    // Vectorized version for array inputs
    vector fourier_series_vector(vector x, vector sin_coeffs, vector cos_coeffs, real period) {
        int n = num_elements(x);
        vector[n] result;
        
        for (i in 1:n) {
        result[i] = fourier_series(x[i], sin_coeffs, cos_coeffs, period);
        }
        
        return result;
    }
    
    // With exponential decay version
    real fourier_exp_decay(real x, real decay_rate, vector sin_coeffs, vector cos_coeffs, real period) {
        real decay = exp(-decay_rate * x);
        real result = 0;
        int n_sin = num_elements(sin_coeffs);
        int n_cos = num_elements(cos_coeffs);
        
        // Add sine components with decay
        for (i in 1:n_sin) {
        result += decay * sin_coeffs[i] * sin(2 * pi() * i * x / period);
        }
        
        // Add cosine components with decay
        for (i in 1:n_cos) {
        result += decay * cos_coeffs[i] * cos(2 * pi() * i * x / period);
        }
        
        return result;
    }
    
    // Vectorized version of exponential decay
    vector fourier_exp_decay_vector(vector x, real decay_rate, vector sin_coeffs, vector cos_coeffs, real period) {
        int n = num_elements(x);
        vector[n] result;
        
        for (i in 1:n) {
        result[i] = fourier_exp_decay(x[i], decay_rate, sin_coeffs, cos_coeffs, period);
        }
        
        return result;
    }
}

// Inputs are log returns; we model latent log variance (log_v)
data {
    int<lower=1> T;              // Number of observations
    vector[T] returns;           // Log returns (observed)
    int<lower=0> T_future;       // Steps to simulate forward

    real quarter_period;         // Trading days in quarter (~63)
    vector<lower=0>[T] day_in_quarter;  // Day within quarter (0-63)

    // For future predictions
    vector<lower=0>[T_future] day_in_quarter_future;  // Future days in quarter
}

transformed data {

    int K = 4; // Number of sin/cos terms in Fourier series

    vector[K] sin_prior = [
        0.001602,  // First sine term
        0.003952,  // Second sine term
        -0.001201,  // Third sine term
        -0.000382   // Fourth sine term
    ]';

    vector[K] cos_prior = [
        0.003364,  // First cos term
        -0.003980,  // Second cos term
        0.000159,  // Third cos term
        0.000786   // Fourth cos term
    ]';

}

parameters {

    real<lower=1,upper=50> nu;   // student t tail

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

    // periodic components for variance
    vector[K] sin_coeffs;
    vector[K] cos_coeffs;
    real<lower=0> decay_rate;

}

transformed parameters {

    // period components
    vector[T] seasonal_effect;
    
    // Calculate seasonal effect for each observation
    seasonal_effect = fourier_exp_decay_vector(
        day_in_quarter,
        decay_rate, 
        sin_coeffs, 
        cos_coeffs, 
        quarter_period
    );

    // rest of model
    vector[T] log_v;
    vector[T] v;
    vector[T] sigma_t;
    vector[T] r_mu;
    vector[T] r_sd;

    real rho = tanh(rho_raw);    // Transform raw correlation to [-1, 1]

    log_v[1] = log_v1;
    for (t in 2:T) {
        real drift = kappa * (log_theta - log_v[t - 1]);
        log_v[t] = log_v[t - 1] + drift + sigma * log_v_std[t - 1] + seasonal_effect[t];
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
    // Priors for seasonal components
    sin_coeffs ~ normal(sin_prior, 0.001);
    cos_coeffs ~ normal(cos_prior, 0.001);
    
    decay_rate ~ normal(0.004600,0.001);  // Prior for decay rate

    nu ~ gamma(2, 0.1); // Prior for student t tail -- this prior is commonly used in practice

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
        returns[t] ~ student_t(nu, r_mu[t], sigma_t[t]);
        target += -0.5 * square(innovation);  // log density of std normal
    }



}

generated quantities {
    vector[T_future] returns_future;
    vector[T_future] log_v_future;
    vector[T_future] v_future;

    vector[T_future] seasonal_effect_future;

    seasonal_effect_future = fourier_exp_decay_vector(
        day_in_quarter_future,
        decay_rate, 
        sin_coeffs, 
        cos_coeffs, 
        quarter_period
    );

    real last_log_v = log_v[T];
    for (t in 1:T_future) {
        real v_prev = exp(last_log_v);
        real next_eps_v = normal_rng(0, 1);
        real next_eps_r = normal_rng(0, 1);
        real next_log_v = last_log_v + kappa * (log_theta - last_log_v) + sigma * next_eps_v + seasonal_effect_future[t];
        log_v_future[t] = next_log_v;
        v_future[t] = exp(next_log_v);

        returns_future[t] = student_t_rng(
                                nu,
                                mu - 0.5 * v_future[t],
                                sqrt(v_future[t])
                                );

        last_log_v = next_log_v;
    }
}
