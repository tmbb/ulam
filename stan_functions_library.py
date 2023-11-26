
from py_expr import PyExpr, python_to_ast
from stan_ast import FunctionCall

class StanFunctionsLibrary:
    def _function(self, function_name, args):
        arguments_as_ast = [python_to_ast(arg) for arg in args]
        function_call_ast = FunctionCall(function_name, arguments_as_ast)
        self.functions[function_name] = self.functions.get(function_name, 0) + 1

        return PyExpr(self, function_call_ast)
              

    def abs(self, z):
        return self._function('abs', [z])          

    def acos(self, z):
        return self._function('acos', [z])          

    def acosh(self, z):
        return self._function('acosh', [z])          

    def add_diag(self, m, d):
        return self._function('add_diag', [m, d])          

    def algebra_solver(self, algebra_system, y_guess, theta, x_r, x_i, rel_tol, f_tol, max_steps):
        return self._function('algebra_solver', [algebra_system, y_guess, theta, x_r, x_i, rel_tol, f_tol, max_steps])          

    def algebra_solver_newton(self, algebra_system, y_guess, theta, x_r, x_i):
        return self._function('algebra_solver_newton', [algebra_system, y_guess, theta, x_r, x_i])          

    def append_array(self, x, y):
        return self._function('append_array', [x, y])          

    def append_col(self, x, y):
        return self._function('append_col', [x, y])          

    def append_row(self, x, y):
        return self._function('append_row', [x, y])          

    def arg(self, z):
        return self._function('arg', [z])          

    def asin(self, z):
        return self._function('asin', [z])          

    def asinh(self, z):
        return self._function('asinh', [z])          

    def atan(self, z):
        return self._function('atan', [z])          

    def atan2(self, y, x):
        return self._function('atan2', [y, x])          

    def atanh(self, z):
        return self._function('atanh', [z])          

    def Bernoulli(self, theta):
        return self._function('bernoulli', [theta])          

    def bernoulli_cdf(self, y, theta):
        return self._function('bernoulli_cdf', [y, theta])          

    def bernoulli_lccdf(self, y, theta):
        return self._function('bernoulli_lccdf', [y, theta])          

    def bernoulli_lcdf(self, y, theta):
        return self._function('bernoulli_lcdf', [y, theta])          

    def BernoulliLogit(self, alpha):
        return self._function('bernoulli_logit', [alpha])          

    def BernoulliLogitGlm(self, x, alpha, beta):
        return self._function('bernoulli_logit_glm', [x, alpha, beta])          

    def bernoulli_logit_glm_lpmf(self, y, x, alpha, beta):
        return self._function('bernoulli_logit_glm_lpmf', [y, x, alpha, beta])          

    def bernoulli_logit_glm_lupmf(self, y, x, alpha, beta):
        return self._function('bernoulli_logit_glm_lupmf', [y, x, alpha, beta])          

    def bernoulli_logit_glm_rng(self, x, alpha, beta):
        return self._function('bernoulli_logit_glm_rng', [x, alpha, beta])          

    def bernoulli_logit_lpmf(self, y, alpha):
        return self._function('bernoulli_logit_lpmf', [y, alpha])          

    def bernoulli_logit_lupmf(self, y, alpha):
        return self._function('bernoulli_logit_lupmf', [y, alpha])          

    def bernoulli_logit_rng(self, alpha):
        return self._function('bernoulli_logit_rng', [alpha])          

    def bernoulli_lpmf(self, y, theta):
        return self._function('bernoulli_lpmf', [y, theta])          

    def bernoulli_lupmf(self, y, theta):
        return self._function('bernoulli_lupmf', [y, theta])          

    def bernoulli_rng(self, theta):
        return self._function('bernoulli_rng', [theta])          

    def bessel_first_kind(self, v, x):
        return self._function('bessel_first_kind', [v, x])          

    def bessel_second_kind(self, v, x):
        return self._function('bessel_second_kind', [v, x])          

    def Beta(self, alpha, beta):
        return self._function('beta', [alpha, beta])          

    def BetaBinomial(self, N, alpha, beta):
        return self._function('beta_binomial', [N, alpha, beta])          

    def beta_binomial_cdf(self, n, N, alpha, beta):
        return self._function('beta_binomial_cdf', [n, N, alpha, beta])          

    def beta_binomial_lccdf(self, n, N, alpha, beta):
        return self._function('beta_binomial_lccdf', [n, N, alpha, beta])          

    def beta_binomial_lcdf(self, n, N, alpha, beta):
        return self._function('beta_binomial_lcdf', [n, N, alpha, beta])          

    def beta_binomial_lpmf(self, n, N, alpha, beta):
        return self._function('beta_binomial_lpmf', [n, N, alpha, beta])          

    def beta_binomial_lupmf(self, n, N, alpha, beta):
        return self._function('beta_binomial_lupmf', [n, N, alpha, beta])          

    def beta_binomial_rng(self, N, alpha, beta):
        return self._function('beta_binomial_rng', [N, alpha, beta])          

    def beta_cdf(self, theta, alpha, beta):
        return self._function('beta_cdf', [theta, alpha, beta])          

    def beta_lccdf(self, theta, alpha, beta):
        return self._function('beta_lccdf', [theta, alpha, beta])          

    def beta_lcdf(self, theta, alpha, beta):
        return self._function('beta_lcdf', [theta, alpha, beta])          

    def beta_lpdf(self, theta, alpha, beta):
        return self._function('beta_lpdf', [theta, alpha, beta])          

    def beta_lupdf(self, theta, alpha, beta):
        return self._function('beta_lupdf', [theta, alpha, beta])          

    def beta_proportion_lccdf(self, theta, mu, kappa):
        return self._function('beta_proportion_lccdf', [theta, mu, kappa])          

    def beta_proportion_lcdf(self, theta, mu, kappa):
        return self._function('beta_proportion_lcdf', [theta, mu, kappa])          

    def beta_proportion_rng(self, mu, kappa):
        return self._function('beta_proportion_rng', [mu, kappa])          

    def beta_rng(self, alpha, beta):
        return self._function('beta_rng', [alpha, beta])          

    def binary_log_loss(self, y, y_hat):
        return self._function('binary_log_loss', [y, y_hat])          

    def Binomial(self, N, theta):
        return self._function('binomial', [N, theta])          

    def binomial_cdf(self, n, N, theta):
        return self._function('binomial_cdf', [n, N, theta])          

    def binomial_lccdf(self, n, N, theta):
        return self._function('binomial_lccdf', [n, N, theta])          

    def binomial_lcdf(self, n, N, theta):
        return self._function('binomial_lcdf', [n, N, theta])          

    def BinomialLogit(self, N, alpha):
        return self._function('binomial_logit', [N, alpha])          

    def binomial_logit_lpmf(self, n, N, alpha):
        return self._function('binomial_logit_lpmf', [n, N, alpha])          

    def binomial_logit_lupmf(self, n, N, alpha):
        return self._function('binomial_logit_lupmf', [n, N, alpha])          

    def binomial_lpmf(self, n, N, theta):
        return self._function('binomial_lpmf', [n, N, theta])          

    def binomial_lupmf(self, n, N, theta):
        return self._function('binomial_lupmf', [n, N, theta])          

    def binomial_rng(self, N, theta):
        return self._function('binomial_rng', [N, theta])          

    def block(self, x, i, j, n_rows, n_cols):
        return self._function('block', [x, i, j, n_rows, n_cols])          

    def Categorical(self, theta):
        return self._function('categorical', [theta])          

    def CategoricalLogit(self, beta):
        return self._function('categorical_logit', [beta])          

    def CategoricalLogitGlm(self, x, alpha, beta):
        return self._function('categorical_logit_glm', [x, alpha, beta])          

    def categorical_logit_glm_lpmf(self, y, x, alpha, beta):
        return self._function('categorical_logit_glm_lpmf', [y, x, alpha, beta])          

    def categorical_logit_glm_lupmf(self, y, x, alpha, beta):
        return self._function('categorical_logit_glm_lupmf', [y, x, alpha, beta])          

    def categorical_logit_lpmf(self, y, beta):
        return self._function('categorical_logit_lpmf', [y, beta])          

    def categorical_logit_lupmf(self, y, beta):
        return self._function('categorical_logit_lupmf', [y, beta])          

    def categorical_logit_rng(self, beta):
        return self._function('categorical_logit_rng', [beta])          

    def categorical_lpmf(self, y, theta):
        return self._function('categorical_lpmf', [y, theta])          

    def categorical_lupmf(self, y, theta):
        return self._function('categorical_lupmf', [y, theta])          

    def categorical_rng(self, theta):
        return self._function('categorical_rng', [theta])          

    def Cauchy(self, mu, sigma):
        return self._function('cauchy', [mu, sigma])          

    def cauchy_cdf(self, y, mu, sigma):
        return self._function('cauchy_cdf', [y, mu, sigma])          

    def cauchy_lccdf(self, y, mu, sigma):
        return self._function('cauchy_lccdf', [y, mu, sigma])          

    def cauchy_lcdf(self, y, mu, sigma):
        return self._function('cauchy_lcdf', [y, mu, sigma])          

    def cauchy_lpdf(self, y, mu, sigma):
        return self._function('cauchy_lpdf', [y, mu, sigma])          

    def cauchy_lupdf(self, y, mu, sigma):
        return self._function('cauchy_lupdf', [y, mu, sigma])          

    def cauchy_rng(self, mu, sigma):
        return self._function('cauchy_rng', [mu, sigma])          

    def cbrt(self, x):
        return self._function('cbrt', [x])          

    def ceil(self, x):
        return self._function('ceil', [x])          

    def ChiSquare(self, nu):
        return self._function('chi_square', [nu])          

    def chi_square_cdf(self, y, nu):
        return self._function('chi_square_cdf', [y, nu])          

    def chi_square_lccdf(self, y, nu):
        return self._function('chi_square_lccdf', [y, nu])          

    def chi_square_lcdf(self, y, nu):
        return self._function('chi_square_lcdf', [y, nu])          

    def chi_square_lpdf(self, y, nu):
        return self._function('chi_square_lpdf', [y, nu])          

    def chi_square_lupdf(self, y, nu):
        return self._function('chi_square_lupdf', [y, nu])          

    def chi_square_rng(self, nu):
        return self._function('chi_square_rng', [nu])          

    def chol2inv(self, L):
        return self._function('chol2inv', [L])          

    def cholesky_decompose(self, A):
        return self._function('cholesky_decompose', [A])          

    def choose(self, x, y):
        return self._function('choose', [x, y])          

    def col(self, x, n):
        return self._function('col', [x, n])          

    def cols(self, x):
        return self._function('cols', [x])          

    def columns_dot_product(self, x, y):
        return self._function('columns_dot_product', [x, y])          

    def columns_dot_self(self, x):
        return self._function('columns_dot_self', [x])          

    def complex_schur_decompose(self, A):
        return self._function('complex_schur_decompose', [A])          

    def complex_schur_decompose_t(self, A):
        return self._function('complex_schur_decompose_t', [A])          

    def complex_schur_decompose_u(self, A):
        return self._function('complex_schur_decompose_u', [A])          

    def conj(self, z):
        return self._function('conj', [z])          

    def cos(self, z):
        return self._function('cos', [z])          

    def cosh(self, z):
        return self._function('cosh', [z])          

    def cov_exp_quad(self, x, alpha, rho):
        return self._function('cov_exp_quad', [x, alpha, rho])          

    def crossprod(self, x):
        return self._function('crossprod', [x])          

    def csr_extract(self, a):
        return self._function('csr_extract', [a])          

    def csr_extract_u(self, a):
        return self._function('csr_extract_u', [a])          

    def csr_extract_v(self, a):
        return self._function('csr_extract_v', [a])          

    def csr_extract_w(self, a):
        return self._function('csr_extract_w', [a])          

    def csr_matrix_times_vector(self, m, n, w, v, u, b):
        return self._function('csr_matrix_times_vector', [m, n, w, v, u, b])          

    def csr_to_dense_matrix(self, m, n, w, v, u):
        return self._function('csr_to_dense_matrix', [m, n, w, v, u])          

    def cumulative_sum(self, x):
        return self._function('cumulative_sum', [x])          

    def determinant(self, A):
        return self._function('determinant', [A])          

    def diag_matrix(self, x):
        return self._function('diag_matrix', [x])          

    def diag_post_multiply(self, m, v):
        return self._function('diag_post_multiply', [m, v])          

    def diag_pre_multiply(self, v, m):
        return self._function('diag_pre_multiply', [v, m])          

    def diagonal(self, x):
        return self._function('diagonal', [x])          

    def digamma(self, x):
        return self._function('digamma', [x])          

    def dims(self, x):
        return self._function('dims', [x])          

    def Dirichlet(self, alpha):
        return self._function('dirichlet', [alpha])          

    def dirichlet_lpdf(self, theta, alpha):
        return self._function('dirichlet_lpdf', [theta, alpha])          

    def dirichlet_lupdf(self, theta, alpha):
        return self._function('dirichlet_lupdf', [theta, alpha])          

    def dirichlet_rng(self, alpha):
        return self._function('dirichlet_rng', [alpha])          

    def DiscreteRange(self, l, u):
        return self._function('discrete_range', [l, u])          

    def discrete_range_cdf(self, y, l, u):
        return self._function('discrete_range_cdf', [y, l, u])          

    def discrete_range_lccdf(self, y, l, u):
        return self._function('discrete_range_lccdf', [y, l, u])          

    def discrete_range_lcdf(self, y, l, u):
        return self._function('discrete_range_lcdf', [y, l, u])          

    def discrete_range_lpmf(self, y, l, u):
        return self._function('discrete_range_lpmf', [y, l, u])          

    def discrete_range_lupmf(self, y, l, u):
        return self._function('discrete_range_lupmf', [y, l, u])          

    def discrete_range_rng(self, l, u):
        return self._function('discrete_range_rng', [l, u])          

    def distance(self, x, y):
        return self._function('distance', [x, y])          

    def dot_product(self, x, y):
        return self._function('dot_product', [x, y])          

    def dot_self(self, x):
        return self._function('dot_self', [x])          

    def DoubleExponential(self, mu, sigma):
        return self._function('double_exponential', [mu, sigma])          

    def double_exponential_cdf(self, y, mu, sigma):
        return self._function('double_exponential_cdf', [y, mu, sigma])          

    def double_exponential_lccdf(self, y, mu, sigma):
        return self._function('double_exponential_lccdf', [y, mu, sigma])          

    def double_exponential_lcdf(self, y, mu, sigma):
        return self._function('double_exponential_lcdf', [y, mu, sigma])          

    def double_exponential_lpdf(self, y, mu, sigma):
        return self._function('double_exponential_lpdf', [y, mu, sigma])          

    def double_exponential_lupdf(self, y, mu, sigma):
        return self._function('double_exponential_lupdf', [y, mu, sigma])          

    def double_exponential_rng(self, mu, sigma):
        return self._function('double_exponential_rng', [mu, sigma])          

    def eigendecompose(self, A):
        return self._function('eigendecompose', [A])          

    def eigendecompose_sym(self, A):
        return self._function('eigendecompose_sym', [A])          

    def eigenvalues(self, A):
        return self._function('eigenvalues', [A])          

    def eigenvalues_sym(self, A):
        return self._function('eigenvalues_sym', [A])          

    def eigenvectors(self, A):
        return self._function('eigenvectors', [A])          

    def eigenvectors_sym(self, A):
        return self._function('eigenvectors_sym', [A])          

    def erf(self, x):
        return self._function('erf', [x])          

    def erfc(self, x):
        return self._function('erfc', [x])          

    def exp(self, z):
        return self._function('exp', [z])          

    def exp2(self, x):
        return self._function('exp2', [x])          

    def ExpModNormal(self, mu, sigma, lambda_):
        return self._function('exp_mod_normal', [mu, sigma, lambda_])          

    def exp_mod_normal_cdf(self, y, mu, sigma, lambda_):
        return self._function('exp_mod_normal_cdf', [y, mu, sigma, lambda_])          

    def exp_mod_normal_lccdf(self, y, mu, sigma, lambda_):
        return self._function('exp_mod_normal_lccdf', [y, mu, sigma, lambda_])          

    def exp_mod_normal_lcdf(self, y, mu, sigma, lambda_):
        return self._function('exp_mod_normal_lcdf', [y, mu, sigma, lambda_])          

    def exp_mod_normal_lpdf(self, y, mu, sigma, lambda_):
        return self._function('exp_mod_normal_lpdf', [y, mu, sigma, lambda_])          

    def exp_mod_normal_lupdf(self, y, mu, sigma, lambda_):
        return self._function('exp_mod_normal_lupdf', [y, mu, sigma, lambda_])          

    def exp_mod_normal_rng(self, mu, sigma, lambda_):
        return self._function('exp_mod_normal_rng', [mu, sigma, lambda_])          

    def expm1(self, x):
        return self._function('expm1', [x])          

    def Exponential(self, beta):
        return self._function('exponential', [beta])          

    def exponential_cdf(self, y, beta):
        return self._function('exponential_cdf', [y, beta])          

    def exponential_lccdf(self, y, beta):
        return self._function('exponential_lccdf', [y, beta])          

    def exponential_lcdf(self, y, beta):
        return self._function('exponential_lcdf', [y, beta])          

    def exponential_lpdf(self, y, beta):
        return self._function('exponential_lpdf', [y, beta])          

    def exponential_lupdf(self, y, beta):
        return self._function('exponential_lupdf', [y, beta])          

    def exponential_rng(self, beta):
        return self._function('exponential_rng', [beta])          

    def falling_factorial(self, x, n):
        return self._function('falling_factorial', [x, n])          

    def fdim(self, x, y):
        return self._function('fdim', [x, y])          

    def fft(self, v):
        return self._function('fft', [v])          

    def fft2(self, m):
        return self._function('fft2', [m])          

    def floor(self, x):
        return self._function('floor', [x])          

    def fma(self, x, y, z):
        return self._function('fma', [x, y, z])          

    def fmax(self, x, y):
        return self._function('fmax', [x, y])          

    def fmin(self, x, y):
        return self._function('fmin', [x, y])          

    def fmod(self, x, y):
        return self._function('fmod', [x, y])          

    def Frechet(self, alpha, sigma):
        return self._function('frechet', [alpha, sigma])          

    def frechet_cdf(self, y, alpha, sigma):
        return self._function('frechet_cdf', [y, alpha, sigma])          

    def frechet_lccdf(self, y, alpha, sigma):
        return self._function('frechet_lccdf', [y, alpha, sigma])          

    def frechet_lcdf(self, y, alpha, sigma):
        return self._function('frechet_lcdf', [y, alpha, sigma])          

    def frechet_lpdf(self, y, alpha, sigma):
        return self._function('frechet_lpdf', [y, alpha, sigma])          

    def frechet_lupdf(self, y, alpha, sigma):
        return self._function('frechet_lupdf', [y, alpha, sigma])          

    def frechet_rng(self, alpha, sigma):
        return self._function('frechet_rng', [alpha, sigma])          

    def Gamma(self, alpha, beta):
        return self._function('gamma', [alpha, beta])          

    def gamma_cdf(self, y, alpha, beta):
        return self._function('gamma_cdf', [y, alpha, beta])          

    def gamma_lccdf(self, y, alpha, beta):
        return self._function('gamma_lccdf', [y, alpha, beta])          

    def gamma_lcdf(self, y, alpha, beta):
        return self._function('gamma_lcdf', [y, alpha, beta])          

    def gamma_lpdf(self, y, alpha, beta):
        return self._function('gamma_lpdf', [y, alpha, beta])          

    def gamma_lupdf(self, y, alpha, beta):
        return self._function('gamma_lupdf', [y, alpha, beta])          

    def gamma_p(self, a, z):
        return self._function('gamma_p', [a, z])          

    def gamma_q(self, a, z):
        return self._function('gamma_q', [a, z])          

    def gamma_rng(self, alpha, beta):
        return self._function('gamma_rng', [alpha, beta])          

    def GaussianDlmObs(self, F, G, V, W, m0, C0):
        return self._function('gaussian_dlm_obs', [F, G, V, W, m0, C0])          

    def gaussian_dlm_obs_lpdf(self, y, F, G, V, W, m0, C0):
        return self._function('gaussian_dlm_obs_lpdf', [y, F, G, V, W, m0, C0])          

    def gaussian_dlm_obs_lupdf(self, y, F, G, V, W, m0, C0):
        return self._function('gaussian_dlm_obs_lupdf', [y, F, G, V, W, m0, C0])          

    def generalized_inverse(self, A):
        return self._function('generalized_inverse', [A])          

    def get_imag(self, z):
        return self._function('get_imag', [z])          

    def get_real(self, z):
        return self._function('get_real', [z])          

    def Gumbel(self, mu, beta):
        return self._function('gumbel', [mu, beta])          

    def gumbel_cdf(self, y, mu, beta):
        return self._function('gumbel_cdf', [y, mu, beta])          

    def gumbel_lccdf(self, y, mu, beta):
        return self._function('gumbel_lccdf', [y, mu, beta])          

    def gumbel_lcdf(self, y, mu, beta):
        return self._function('gumbel_lcdf', [y, mu, beta])          

    def gumbel_lpdf(self, y, mu, beta):
        return self._function('gumbel_lpdf', [y, mu, beta])          

    def gumbel_lupdf(self, y, mu, beta):
        return self._function('gumbel_lupdf', [y, mu, beta])          

    def gumbel_rng(self, mu, beta):
        return self._function('gumbel_rng', [mu, beta])          

    def head(self, v, n):
        return self._function('head', [v, n])          

    def hmm_hidden_state_prob(self, log_omega, Gamma, rho):
        return self._function('hmm_hidden_state_prob', [log_omega, Gamma, rho])          

    def hmm_latent_rng(self, log_omega, Gamma, rho):
        return self._function('hmm_latent_rng', [log_omega, Gamma, rho])          

    def hmm_marginal(self, log_omega, Gamma, rho):
        return self._function('hmm_marginal', [log_omega, Gamma, rho])          

    def Hypergeometric(self, N, a, b):
        return self._function('hypergeometric', [N, a, b])          

    def hypergeometric_lpmf(self, n, N, a, b):
        return self._function('hypergeometric_lpmf', [n, N, a, b])          

    def hypergeometric_lupmf(self, n, N, a, b):
        return self._function('hypergeometric_lupmf', [n, N, a, b])          

    def hypergeometric_rng(self, N, a, b):
        return self._function('hypergeometric_rng', [N, a, b])          

    def hypot(self, x, y):
        return self._function('hypot', [x, y])          

    def identity_matrix(self, k):
        return self._function('identity_matrix', [k])          

    def inc_beta(self, alpha, beta, x):
        return self._function('inc_beta', [alpha, beta, x])          

    def int_step(self, x):
        return self._function('int_step', [x])          

    def integrate_1d(self, integrand, a, b, theta, x_r, x_i):
        return self._function('integrate_1d', [integrand, a, b, theta, x_r, x_i])          

    def integrate_ode(self, ode, initial_state, initial_time, times, theta, x_r, x_i):
        return self._function('integrate_ode', [ode, initial_state, initial_time, times, theta, x_r, x_i])          

    def integrate_ode_adams(self, ode, initial_state, initial_time, times, theta, x_r, x_i):
        return self._function('integrate_ode_adams', [ode, initial_state, initial_time, times, theta, x_r, x_i])          

    def integrate_ode_bdf(self, ode, initial_state, initial_time, times, theta, x_r, x_i):
        return self._function('integrate_ode_bdf', [ode, initial_state, initial_time, times, theta, x_r, x_i])          

    def integrate_ode_rk45(self, ode, initial_state, initial_time, times, theta, x_r, x_i):
        return self._function('integrate_ode_rk45', [ode, initial_state, initial_time, times, theta, x_r, x_i])          

    def inv(self, x):
        return self._function('inv', [x])          

    def InvChiSquare(self, nu):
        return self._function('inv_chi_square', [nu])          

    def inv_chi_square_cdf(self, y, nu):
        return self._function('inv_chi_square_cdf', [y, nu])          

    def inv_chi_square_lccdf(self, y, nu):
        return self._function('inv_chi_square_lccdf', [y, nu])          

    def inv_chi_square_lcdf(self, y, nu):
        return self._function('inv_chi_square_lcdf', [y, nu])          

    def inv_chi_square_lpdf(self, y, nu):
        return self._function('inv_chi_square_lpdf', [y, nu])          

    def inv_chi_square_lupdf(self, y, nu):
        return self._function('inv_chi_square_lupdf', [y, nu])          

    def inv_chi_square_rng(self, nu):
        return self._function('inv_chi_square_rng', [nu])          

    def inv_cloglog(self, x):
        return self._function('inv_cloglog', [x])          

    def inv_erfc(self, x):
        return self._function('inv_erfc', [x])          

    def inv_fft(self, u):
        return self._function('inv_fft', [u])          

    def inv_fft2(self, m):
        return self._function('inv_fft2', [m])          

    def InvGamma(self, alpha, beta):
        return self._function('inv_gamma', [alpha, beta])          

    def inv_gamma_cdf(self, y, alpha, beta):
        return self._function('inv_gamma_cdf', [y, alpha, beta])          

    def inv_gamma_lccdf(self, y, alpha, beta):
        return self._function('inv_gamma_lccdf', [y, alpha, beta])          

    def inv_gamma_lcdf(self, y, alpha, beta):
        return self._function('inv_gamma_lcdf', [y, alpha, beta])          

    def inv_gamma_lpdf(self, y, alpha, beta):
        return self._function('inv_gamma_lpdf', [y, alpha, beta])          

    def inv_gamma_lupdf(self, y, alpha, beta):
        return self._function('inv_gamma_lupdf', [y, alpha, beta])          

    def inv_gamma_rng(self, alpha, beta):
        return self._function('inv_gamma_rng', [alpha, beta])          

    def inv_inc_beta(self, alpha, beta, p):
        return self._function('inv_inc_beta', [alpha, beta, p])          

    def inv_logit(self, x):
        return self._function('inv_logit', [x])          

    def inv_Phi(self, x):
        return self._function('inv_Phi', [x])          

    def inv_sqrt(self, x):
        return self._function('inv_sqrt', [x])          

    def inv_square(self, x):
        return self._function('inv_square', [x])          

    def InvWishart(self, nu, Sigma):
        return self._function('inv_wishart', [nu, Sigma])          

    def InvWishartCholesky(self, nu, L_S):
        return self._function('inv_wishart_cholesky', [nu, L_S])          

    def inv_wishart_cholesky_lpdf(self, L_W, nu, L_S):
        return self._function('inv_wishart_cholesky_lpdf', [L_W, nu, L_S])          

    def inv_wishart_cholesky_lupdf(self, L_W, nu, L_S):
        return self._function('inv_wishart_cholesky_lupdf', [L_W, nu, L_S])          

    def inv_wishart_cholesky_rng(self, nu, L_S):
        return self._function('inv_wishart_cholesky_rng', [nu, L_S])          

    def inv_wishart_lpdf(self, W, nu, Sigma):
        return self._function('inv_wishart_lpdf', [W, nu, Sigma])          

    def inv_wishart_lupdf(self, W, nu, Sigma):
        return self._function('inv_wishart_lupdf', [W, nu, Sigma])          

    def inv_wishart_rng(self, nu, Sigma):
        return self._function('inv_wishart_rng', [nu, Sigma])          

    def inverse(self, A):
        return self._function('inverse', [A])          

    def inverse_spd(self, A):
        return self._function('inverse_spd', [A])          

    def is_inf(self, x):
        return self._function('is_inf', [x])          

    def is_nan(self, x):
        return self._function('is_nan', [x])          

    def lambert_w0(self, x):
        return self._function('lambert_w0', [x])          

    def lambert_wm1(self, x):
        return self._function('lambert_wm1', [x])          

    def lbeta(self, alpha, beta):
        return self._function('lbeta', [alpha, beta])          

    def lchoose(self, x, y):
        return self._function('lchoose', [x, y])          

    def ldexp(self, x, y):
        return self._function('ldexp', [x, y])          

    def lgamma(self, x):
        return self._function('lgamma', [x])          

    def linspaced_array(self, n, lower, upper):
        return self._function('linspaced_array', [n, lower, upper])          

    def linspaced_int_array(self, n, lower, upper):
        return self._function('linspaced_int_array', [n, lower, upper])          

    def linspaced_row_vector(self, n, lower, upper):
        return self._function('linspaced_row_vector', [n, lower, upper])          

    def linspaced_vector(self, n, lower, upper):
        return self._function('linspaced_vector', [n, lower, upper])          

    def LkjCorr(self, eta):
        return self._function('lkj_corr', [eta])          

    def LkjCorrCholesky(self, eta):
        return self._function('lkj_corr_cholesky', [eta])          

    def lkj_corr_cholesky_lpdf(self, L, eta):
        return self._function('lkj_corr_cholesky_lpdf', [L, eta])          

    def lkj_corr_cholesky_lupdf(self, L, eta):
        return self._function('lkj_corr_cholesky_lupdf', [L, eta])          

    def lkj_corr_cholesky_rng(self, K, eta):
        return self._function('lkj_corr_cholesky_rng', [K, eta])          

    def lkj_corr_lpdf(self, y, eta):
        return self._function('lkj_corr_lpdf', [y, eta])          

    def lkj_corr_lupdf(self, y, eta):
        return self._function('lkj_corr_lupdf', [y, eta])          

    def lkj_corr_rng(self, K, eta):
        return self._function('lkj_corr_rng', [K, eta])          

    def lmgamma(self, n, x):
        return self._function('lmgamma', [n, x])          

    def lmultiply(self, x, y):
        return self._function('lmultiply', [x, y])          

    def log(self, z):
        return self._function('log', [z])          

    def log10(self, z):
        return self._function('log10', [z])          

    def log1m(self, x):
        return self._function('log1m', [x])          

    def log1m_exp(self, x):
        return self._function('log1m_exp', [x])          

    def log1m_inv_logit(self, x):
        return self._function('log1m_inv_logit', [x])          

    def log1p(self, x):
        return self._function('log1p', [x])          

    def log1p_exp(self, x):
        return self._function('log1p_exp', [x])          

    def log2(self, x):
        return self._function('log2', [x])          

    def log_determinant(self, A):
        return self._function('log_determinant', [A])          

    def log_diff_exp(self, x, y):
        return self._function('log_diff_exp', [x, y])          

    def log_falling_factorial(self, x, n):
        return self._function('log_falling_factorial', [x, n])          

    def log_inv_logit(self, x):
        return self._function('log_inv_logit', [x])          

    def log_inv_logit_diff(self, x, y):
        return self._function('log_inv_logit_diff', [x, y])          

    def log_mix(self, theta, lp1, lp2):
        return self._function('log_mix', [theta, lp1, lp2])          

    def log_modified_bessel_first_kind(self, v, z):
        return self._function('log_modified_bessel_first_kind', [v, z])          

    def log_rising_factorial(self, x, n):
        return self._function('log_rising_factorial', [x, n])          

    def log_softmax(self, x):
        return self._function('log_softmax', [x])          

    def log_sum_exp(self, x):
        return self._function('log_sum_exp', [x])          

    def Logistic(self, mu, sigma):
        return self._function('logistic', [mu, sigma])          

    def logistic_cdf(self, y, mu, sigma):
        return self._function('logistic_cdf', [y, mu, sigma])          

    def logistic_lccdf(self, y, mu, sigma):
        return self._function('logistic_lccdf', [y, mu, sigma])          

    def logistic_lcdf(self, y, mu, sigma):
        return self._function('logistic_lcdf', [y, mu, sigma])          

    def logistic_lpdf(self, y, mu, sigma):
        return self._function('logistic_lpdf', [y, mu, sigma])          

    def logistic_lupdf(self, y, mu, sigma):
        return self._function('logistic_lupdf', [y, mu, sigma])          

    def logistic_rng(self, mu, sigma):
        return self._function('logistic_rng', [mu, sigma])          

    def logit(self, x):
        return self._function('logit', [x])          

    def Loglogistic(self, alpha, beta):
        return self._function('loglogistic', [alpha, beta])          

    def loglogistic_cdf(self, y, alpha, beta):
        return self._function('loglogistic_cdf', [y, alpha, beta])          

    def loglogistic_lpdf(self, y, alpha, beta):
        return self._function('loglogistic_lpdf', [y, alpha, beta])          

    def loglogistic_rng(self, mu, sigma):
        return self._function('loglogistic_rng', [mu, sigma])          

    def Lognormal(self, mu, sigma):
        return self._function('lognormal', [mu, sigma])          

    def lognormal_cdf(self, y, mu, sigma):
        return self._function('lognormal_cdf', [y, mu, sigma])          

    def lognormal_lccdf(self, y, mu, sigma):
        return self._function('lognormal_lccdf', [y, mu, sigma])          

    def lognormal_lcdf(self, y, mu, sigma):
        return self._function('lognormal_lcdf', [y, mu, sigma])          

    def lognormal_lpdf(self, y, mu, sigma):
        return self._function('lognormal_lpdf', [y, mu, sigma])          

    def lognormal_lupdf(self, y, mu, sigma):
        return self._function('lognormal_lupdf', [y, mu, sigma])          

    def lognormal_rng(self, mu, sigma):
        return self._function('lognormal_rng', [mu, sigma])          

    def matrix_exp(self, A):
        return self._function('matrix_exp', [A])          

    def matrix_exp_multiply(self, A, B):
        return self._function('matrix_exp_multiply', [A, B])          

    def matrix_power(self, A, B):
        return self._function('matrix_power', [A, B])          

    def max(self, x):
        return self._function('max', [x])          

    def mdivide_left_spd(self, A, b):
        return self._function('mdivide_left_spd', [A, b])          

    def mdivide_left_tri_low(self, A, b):
        return self._function('mdivide_left_tri_low', [A, b])          

    def mdivide_right_spd(self, b, A):
        return self._function('mdivide_right_spd', [b, A])          

    def mdivide_right_tri_low(self, b, A):
        return self._function('mdivide_right_tri_low', [b, A])          

    def mean(self, x):
        return self._function('mean', [x])          

    def min(self, x):
        return self._function('min', [x])          

    def modified_bessel_first_kind(self, v, z):
        return self._function('modified_bessel_first_kind', [v, z])          

    def modified_bessel_second_kind(self, v, z):
        return self._function('modified_bessel_second_kind', [v, z])          

    def MultiGp(self, Sigma, w):
        return self._function('multi_gp', [Sigma, w])          

    def MultiGpCholesky(self, L, w):
        return self._function('multi_gp_cholesky', [L, w])          

    def multi_gp_cholesky_lpdf(self, y, L, w):
        return self._function('multi_gp_cholesky_lpdf', [y, L, w])          

    def multi_gp_cholesky_lupdf(self, y, L, w):
        return self._function('multi_gp_cholesky_lupdf', [y, L, w])          

    def multi_gp_lpdf(self, y, Sigma, w):
        return self._function('multi_gp_lpdf', [y, Sigma, w])          

    def multi_gp_lupdf(self, y, Sigma, w):
        return self._function('multi_gp_lupdf', [y, Sigma, w])          

    def MultiNormal(self, mu, Sigma):
        return self._function('multi_normal', [mu, Sigma])          

    def MultiNormalCholesky(self, mu, L):
        return self._function('multi_normal_cholesky', [mu, L])          

    def multi_normal_cholesky_lpdf(self, y, mu, L):
        return self._function('multi_normal_cholesky_lpdf', [y, mu, L])          

    def multi_normal_cholesky_lupdf(self, y, mu, L):
        return self._function('multi_normal_cholesky_lupdf', [y, mu, L])          

    def multi_normal_cholesky_rng(self, mu, L):
        return self._function('multi_normal_cholesky_rng', [mu, L])          

    def multi_normal_lpdf(self, y, mu, Sigma):
        return self._function('multi_normal_lpdf', [y, mu, Sigma])          

    def multi_normal_lupdf(self, y, mu, Sigma):
        return self._function('multi_normal_lupdf', [y, mu, Sigma])          

    def MultiNormalPrec(self, mu, Omega):
        return self._function('multi_normal_prec', [mu, Omega])          

    def multi_normal_prec_lpdf(self, y, mu, Omega):
        return self._function('multi_normal_prec_lpdf', [y, mu, Omega])          

    def multi_normal_prec_lupdf(self, y, mu, Omega):
        return self._function('multi_normal_prec_lupdf', [y, mu, Omega])          

    def multi_normal_rng(self, mu, Sigma):
        return self._function('multi_normal_rng', [mu, Sigma])          

    def multi_student_cholesky_t_rng(self, nu, mu, L):
        return self._function('multi_student_cholesky_t_rng', [nu, mu, L])          

    def MultiStudentT(self, nu, mu, Sigma):
        return self._function('multi_student_t', [nu, mu, Sigma])          

    def MultiStudentTCholesky(self, nu, mu, L):
        return self._function('multi_student_t_cholesky', [nu, mu, L])          

    def multi_student_t_cholesky_lpdf(self, y, nu, mu, L):
        return self._function('multi_student_t_cholesky_lpdf', [y, nu, mu, L])          

    def multi_student_t_cholesky_lupdf(self, y, nu, mu, L):
        return self._function('multi_student_t_cholesky_lupdf', [y, nu, mu, L])          

    def multi_student_t_cholesky_rng(self, nu, mu, L):
        return self._function('multi_student_t_cholesky_rng', [nu, mu, L])          

    def multi_student_t_lpdf(self, y, nu, mu, Sigma):
        return self._function('multi_student_t_lpdf', [y, nu, mu, Sigma])          

    def multi_student_t_lupdf(self, y, nu, mu, Sigma):
        return self._function('multi_student_t_lupdf', [y, nu, mu, Sigma])          

    def multi_student_t_rng(self, nu, mu, Sigma):
        return self._function('multi_student_t_rng', [nu, mu, Sigma])          

    def Multinomial(self, theta):
        return self._function('multinomial', [theta])          

    def MultinomialLogit(self, gamma):
        return self._function('multinomial_logit', [gamma])          

    def multinomial_logit_lpmf(self, y, gamma):
        return self._function('multinomial_logit_lpmf', [y, gamma])          

    def multinomial_logit_lupmf(self, y, gamma):
        return self._function('multinomial_logit_lupmf', [y, gamma])          

    def multinomial_logit_rng(self, gamma, N):
        return self._function('multinomial_logit_rng', [gamma, N])          

    def multinomial_lpmf(self, y, theta):
        return self._function('multinomial_lpmf', [y, theta])          

    def multinomial_lupmf(self, y, theta):
        return self._function('multinomial_lupmf', [y, theta])          

    def multinomial_rng(self, theta, N):
        return self._function('multinomial_rng', [theta, N])          

    def multiply_lower_tri_self_transpose(self, x):
        return self._function('multiply_lower_tri_self_transpose', [x])          

    def NegBinomial(self, alpha, beta):
        return self._function('neg_binomial', [alpha, beta])          

    def NegBinomial2(self, mu, phi):
        return self._function('neg_binomial_2', [mu, phi])          

    def neg_binomial_2_cdf(self, n, mu, phi):
        return self._function('neg_binomial_2_cdf', [n, mu, phi])          

    def neg_binomial_2_lccdf(self, n, mu, phi):
        return self._function('neg_binomial_2_lccdf', [n, mu, phi])          

    def neg_binomial_2_lcdf(self, n, mu, phi):
        return self._function('neg_binomial_2_lcdf', [n, mu, phi])          

    def NegBinomial2Log(self, eta, phi):
        return self._function('neg_binomial_2_log', [eta, phi])          

    def NegBinomial2LogGlm(self, x, alpha, beta, phi):
        return self._function('neg_binomial_2_log_glm', [x, alpha, beta, phi])          

    def neg_binomial_2_log_glm_lpmf(self, y, x, alpha, beta, phi):
        return self._function('neg_binomial_2_log_glm_lpmf', [y, x, alpha, beta, phi])          

    def neg_binomial_2_log_glm_lupmf(self, y, x, alpha, beta, phi):
        return self._function('neg_binomial_2_log_glm_lupmf', [y, x, alpha, beta, phi])          

    def neg_binomial_2_log_lpmf(self, n, eta, phi):
        return self._function('neg_binomial_2_log_lpmf', [n, eta, phi])          

    def neg_binomial_2_log_lupmf(self, n, eta, phi):
        return self._function('neg_binomial_2_log_lupmf', [n, eta, phi])          

    def neg_binomial_2_log_rng(self, eta, phi):
        return self._function('neg_binomial_2_log_rng', [eta, phi])          

    def neg_binomial_2_lpmf(self, n, mu, phi):
        return self._function('neg_binomial_2_lpmf', [n, mu, phi])          

    def neg_binomial_2_lupmf(self, n, mu, phi):
        return self._function('neg_binomial_2_lupmf', [n, mu, phi])          

    def neg_binomial_2_rng(self, mu, phi):
        return self._function('neg_binomial_2_rng', [mu, phi])          

    def neg_binomial_cdf(self, n, alpha, beta):
        return self._function('neg_binomial_cdf', [n, alpha, beta])          

    def neg_binomial_lccdf(self, n, alpha, beta):
        return self._function('neg_binomial_lccdf', [n, alpha, beta])          

    def neg_binomial_lcdf(self, n, alpha, beta):
        return self._function('neg_binomial_lcdf', [n, alpha, beta])          

    def neg_binomial_lpmf(self, n, alpha, beta):
        return self._function('neg_binomial_lpmf', [n, alpha, beta])          

    def neg_binomial_lupmf(self, n, alpha, beta):
        return self._function('neg_binomial_lupmf', [n, alpha, beta])          

    def neg_binomial_rng(self, alpha, beta):
        return self._function('neg_binomial_rng', [alpha, beta])          

    def norm(self, z):
        return self._function('norm', [z])          

    def norm1(self, x):
        return self._function('norm1', [x])          

    def norm2(self, x):
        return self._function('norm2', [x])          

    def Normal(self, mu, sigma):
        return self._function('normal', [mu, sigma])          

    def normal_cdf(self, y, mu, sigma):
        return self._function('normal_cdf', [y, mu, sigma])          

    def NormalIdGlm(self, x, alpha, beta, sigma):
        return self._function('normal_id_glm', [x, alpha, beta, sigma])          

    def normal_id_glm_lpdf(self, y, x, alpha, beta, sigma):
        return self._function('normal_id_glm_lpdf', [y, x, alpha, beta, sigma])          

    def normal_id_glm_lupdf(self, y, x, alpha, beta, sigma):
        return self._function('normal_id_glm_lupdf', [y, x, alpha, beta, sigma])          

    def normal_lccdf(self, y, mu, sigma):
        return self._function('normal_lccdf', [y, mu, sigma])          

    def normal_lcdf(self, y, mu, sigma):
        return self._function('normal_lcdf', [y, mu, sigma])          

    def normal_lpdf(self, y, mu, sigma):
        return self._function('normal_lpdf', [y, mu, sigma])          

    def normal_lupdf(self, y, mu, sigma):
        return self._function('normal_lupdf', [y, mu, sigma])          

    def normal_rng(self, mu, sigma):
        return self._function('normal_rng', [mu, sigma])          

    def num_elements(self, x):
        return self._function('num_elements', [x])          

    def one_hot_array(self, n, k):
        return self._function('one_hot_array', [n, k])          

    def one_hot_int_array(self, n, k):
        return self._function('one_hot_int_array', [n, k])          

    def one_hot_row_vector(self, n, k):
        return self._function('one_hot_row_vector', [n, k])          

    def one_hot_vector(self, K, k):
        return self._function('one_hot_vector', [K, k])          

    def ones_array(self, n):
        return self._function('ones_array', [n])          

    def ones_int_array(self, n):
        return self._function('ones_int_array', [n])          

    def ones_row_vector(self, n):
        return self._function('ones_row_vector', [n])          

    def ones_vector(self, n):
        return self._function('ones_vector', [n])          

    def OrderedLogistic(self, eta, c):
        return self._function('ordered_logistic', [eta, c])          

    def OrderedLogisticGlm(self, x, beta, c):
        return self._function('ordered_logistic_glm', [x, beta, c])          

    def ordered_logistic_glm_lpmf(self, y, x, beta, c):
        return self._function('ordered_logistic_glm_lpmf', [y, x, beta, c])          

    def ordered_logistic_glm_lupmf(self, y, x, beta, c):
        return self._function('ordered_logistic_glm_lupmf', [y, x, beta, c])          

    def ordered_logistic_lpmf(self, k, eta, c):
        return self._function('ordered_logistic_lpmf', [k, eta, c])          

    def ordered_logistic_lupmf(self, k, eta, c):
        return self._function('ordered_logistic_lupmf', [k, eta, c])          

    def ordered_logistic_rng(self, eta, c):
        return self._function('ordered_logistic_rng', [eta, c])          

    def OrderedProbit(self, eta, c):
        return self._function('ordered_probit', [eta, c])          

    def ordered_probit_lpmf(self, k, eta, c):
        return self._function('ordered_probit_lpmf', [k, eta, c])          

    def ordered_probit_lupmf(self, k, eta, c):
        return self._function('ordered_probit_lupmf', [k, eta, c])          

    def ordered_probit_rng(self, eta, c):
        return self._function('ordered_probit_rng', [eta, c])          

    def owens_t(self, h, a):
        return self._function('owens_t', [h, a])          

    def Pareto(self, y_min, alpha):
        return self._function('pareto', [y_min, alpha])          

    def pareto_cdf(self, y, y_min, alpha):
        return self._function('pareto_cdf', [y, y_min, alpha])          

    def pareto_lccdf(self, y, y_min, alpha):
        return self._function('pareto_lccdf', [y, y_min, alpha])          

    def pareto_lcdf(self, y, y_min, alpha):
        return self._function('pareto_lcdf', [y, y_min, alpha])          

    def pareto_lpdf(self, y, y_min, alpha):
        return self._function('pareto_lpdf', [y, y_min, alpha])          

    def pareto_lupdf(self, y, y_min, alpha):
        return self._function('pareto_lupdf', [y, y_min, alpha])          

    def pareto_rng(self, y_min, alpha):
        return self._function('pareto_rng', [y_min, alpha])          

    def ParetoType2(self, mu, lambda_, alpha):
        return self._function('pareto_type_2', [mu, lambda_, alpha])          

    def pareto_type_2_cdf(self, y, mu, lambda_, alpha):
        return self._function('pareto_type_2_cdf', [y, mu, lambda_, alpha])          

    def pareto_type_2_lccdf(self, y, mu, lambda_, alpha):
        return self._function('pareto_type_2_lccdf', [y, mu, lambda_, alpha])          

    def pareto_type_2_lcdf(self, y, mu, lambda_, alpha):
        return self._function('pareto_type_2_lcdf', [y, mu, lambda_, alpha])          

    def pareto_type_2_lpdf(self, y, mu, lambda_, alpha):
        return self._function('pareto_type_2_lpdf', [y, mu, lambda_, alpha])          

    def pareto_type_2_lupdf(self, y, mu, lambda_, alpha):
        return self._function('pareto_type_2_lupdf', [y, mu, lambda_, alpha])          

    def pareto_type_2_rng(self, mu, lambda_, alpha):
        return self._function('pareto_type_2_rng', [mu, lambda_, alpha])          

    def Phi(self, x):
        return self._function('Phi', [x])          

    def Phi_approx(self, x):
        return self._function('Phi_approx', [x])          

    def Poisson(self, lambda_):
        return self._function('poisson', [lambda_])          

    def poisson_cdf(self, n, lambda_):
        return self._function('poisson_cdf', [n, lambda_])          

    def poisson_lccdf(self, n, lambda_):
        return self._function('poisson_lccdf', [n, lambda_])          

    def poisson_lcdf(self, n, lambda_):
        return self._function('poisson_lcdf', [n, lambda_])          

    def PoissonLog(self, alpha):
        return self._function('poisson_log', [alpha])          

    def PoissonLogGlm(self, x, alpha, beta):
        return self._function('poisson_log_glm', [x, alpha, beta])          

    def poisson_log_glm_lpmf(self, y, x, alpha, beta):
        return self._function('poisson_log_glm_lpmf', [y, x, alpha, beta])          

    def poisson_log_glm_lupmf(self, y, x, alpha, beta):
        return self._function('poisson_log_glm_lupmf', [y, x, alpha, beta])          

    def poisson_log_lpmf(self, n, alpha):
        return self._function('poisson_log_lpmf', [n, alpha])          

    def poisson_log_lupmf(self, n, alpha):
        return self._function('poisson_log_lupmf', [n, alpha])          

    def poisson_log_rng(self, alpha):
        return self._function('poisson_log_rng', [alpha])          

    def poisson_lpmf(self, n, lambda_):
        return self._function('poisson_lpmf', [n, lambda_])          

    def poisson_lupmf(self, n, lambda_):
        return self._function('poisson_lupmf', [n, lambda_])          

    def poisson_rng(self, lambda_):
        return self._function('poisson_rng', [lambda_])          

    def polar(self, r, theta):
        return self._function('polar', [r, theta])          

    def pow(self, x, y):
        return self._function('pow', [x, y])          

    def prod(self, x):
        return self._function('prod', [x])          

    def proj(self, z):
        return self._function('proj', [z])          

    def qr(self, A):
        return self._function('qr', [A])          

    def qr_Q(self, A):
        return self._function('qr_Q', [A])          

    def qr_R(self, A):
        return self._function('qr_R', [A])          

    def qr_thin(self, A):
        return self._function('qr_thin', [A])          

    def qr_thin_Q(self, A):
        return self._function('qr_thin_Q', [A])          

    def qr_thin_R(self, A):
        return self._function('qr_thin_R', [A])          

    def quad_form(self, A, B):
        return self._function('quad_form', [A, B])          

    def quad_form_diag(self, m, v):
        return self._function('quad_form_diag', [m, v])          

    def quad_form_sym(self, A, B):
        return self._function('quad_form_sym', [A, B])          

    def quantile(self, x, p):
        return self._function('quantile', [x, p])          

    def rank(self, v, s):
        return self._function('rank', [v, s])          

    def Rayleigh(self, sigma):
        return self._function('rayleigh', [sigma])          

    def rayleigh_cdf(self, y, sigma):
        return self._function('rayleigh_cdf', [y, sigma])          

    def rayleigh_lccdf(self, y, sigma):
        return self._function('rayleigh_lccdf', [y, sigma])          

    def rayleigh_lcdf(self, y, sigma):
        return self._function('rayleigh_lcdf', [y, sigma])          

    def rayleigh_lpdf(self, y, sigma):
        return self._function('rayleigh_lpdf', [y, sigma])          

    def rayleigh_lupdf(self, y, sigma):
        return self._function('rayleigh_lupdf', [y, sigma])          

    def rayleigh_rng(self, sigma):
        return self._function('rayleigh_rng', [sigma])          

    def rep_array(self, x, n):
        return self._function('rep_array', [x, n])          

    def rep_matrix(self, z, m, n):
        return self._function('rep_matrix', [z, m, n])          

    def rep_row_vector(self, z, n):
        return self._function('rep_row_vector', [z, n])          

    def rep_vector(self, z, m):
        return self._function('rep_vector', [z, m])          

    def reverse(self, v):
        return self._function('reverse', [v])          

    def rising_factorial(self, x, n):
        return self._function('rising_factorial', [x, n])          

    def round(self, x):
        return self._function('round', [x])          

    def row(self, x, m):
        return self._function('row', [x, m])          

    def rows(self, x):
        return self._function('rows', [x])          

    def rows_dot_product(self, x, y):
        return self._function('rows_dot_product', [x, y])          

    def rows_dot_self(self, x):
        return self._function('rows_dot_self', [x])          

    def scale_matrix_exp_multiply(self, t, A, B):
        return self._function('scale_matrix_exp_multiply', [t, A, B])          

    def ScaledInvChiSquare(self, nu, sigma):
        return self._function('scaled_inv_chi_square', [nu, sigma])          

    def scaled_inv_chi_square_cdf(self, y, nu, sigma):
        return self._function('scaled_inv_chi_square_cdf', [y, nu, sigma])          

    def scaled_inv_chi_square_lccdf(self, y, nu, sigma):
        return self._function('scaled_inv_chi_square_lccdf', [y, nu, sigma])          

    def scaled_inv_chi_square_lcdf(self, y, nu, sigma):
        return self._function('scaled_inv_chi_square_lcdf', [y, nu, sigma])          

    def scaled_inv_chi_square_lpdf(self, y, nu, sigma):
        return self._function('scaled_inv_chi_square_lpdf', [y, nu, sigma])          

    def scaled_inv_chi_square_lupdf(self, y, nu, sigma):
        return self._function('scaled_inv_chi_square_lupdf', [y, nu, sigma])          

    def scaled_inv_chi_square_rng(self, nu, sigma):
        return self._function('scaled_inv_chi_square_rng', [nu, sigma])          

    def sd(self, x):
        return self._function('sd', [x])          

    def segment(self, v, i, n):
        return self._function('segment', [v, i, n])          

    def sin(self, z):
        return self._function('sin', [z])          

    def singular_values(self, A):
        return self._function('singular_values', [A])          

    def sinh(self, z):
        return self._function('sinh', [z])          

    def size(self, x):
        return self._function('size', [x])          

    def SkewDoubleExponential(self, mu, sigma, tau):
        return self._function('skew_double_exponential', [mu, sigma, tau])          

    def skew_double_exponential_cdf(self, y, mu, sigma, tau):
        return self._function('skew_double_exponential_cdf', [y, mu, sigma, tau])          

    def skew_double_exponential_lccdf(self, y, mu, sigma, tau):
        return self._function('skew_double_exponential_lccdf', [y, mu, sigma, tau])          

    def skew_double_exponential_lcdf(self, y, mu, sigma, tau):
        return self._function('skew_double_exponential_lcdf', [y, mu, sigma, tau])          

    def skew_double_exponential_lpdf(self, y, mu, sigma, tau):
        return self._function('skew_double_exponential_lpdf', [y, mu, sigma, tau])          

    def skew_double_exponential_lupdf(self, y, mu, sigma, tau):
        return self._function('skew_double_exponential_lupdf', [y, mu, sigma, tau])          

    def skew_double_exponential_rng(self, mu, sigma):
        return self._function('skew_double_exponential_rng', [mu, sigma])          

    def SkewNormal(self, xi, omega, alpha):
        return self._function('skew_normal', [xi, omega, alpha])          

    def skew_normal_cdf(self, y, xi, omega, alpha):
        return self._function('skew_normal_cdf', [y, xi, omega, alpha])          

    def skew_normal_lccdf(self, y, xi, omega, alpha):
        return self._function('skew_normal_lccdf', [y, xi, omega, alpha])          

    def skew_normal_lcdf(self, y, xi, omega, alpha):
        return self._function('skew_normal_lcdf', [y, xi, omega, alpha])          

    def skew_normal_lpdf(self, y, xi, omega, alpha):
        return self._function('skew_normal_lpdf', [y, xi, omega, alpha])          

    def skew_normal_lupdf(self, y, xi, omega, alpha):
        return self._function('skew_normal_lupdf', [y, xi, omega, alpha])          

    def skew_normal_rng(self, xi, omega, alpha):
        return self._function('skew_normal_rng', [xi, omega, alpha])          

    def softmax(self, x):
        return self._function('softmax', [x])          

    def sort_asc(self, v):
        return self._function('sort_asc', [v])          

    def sort_desc(self, v):
        return self._function('sort_desc', [v])          

    def sort_indices_asc(self, v):
        return self._function('sort_indices_asc', [v])          

    def sort_indices_desc(self, v):
        return self._function('sort_indices_desc', [v])          

    def sqrt(self, x):
        return self._function('sqrt', [x])          

    def square(self, x):
        return self._function('square', [x])          

    def squared_distance(self, x, y):
        return self._function('squared_distance', [x, y])          

    def StdNormal(self):
        return self._function('std_normal', [])          

    def std_normal_cdf(self, y):
        return self._function('std_normal_cdf', [y])          

    def std_normal_lccdf(self, y):
        return self._function('std_normal_lccdf', [y])          

    def std_normal_lcdf(self, y):
        return self._function('std_normal_lcdf', [y])          

    def std_normal_log_qf(self, x):
        return self._function('std_normal_log_qf', [x])          

    def std_normal_lpdf(self, y):
        return self._function('std_normal_lpdf', [y])          

    def std_normal_lupdf(self, y):
        return self._function('std_normal_lupdf', [y])          

    def std_normal_qf(self, x):
        return self._function('std_normal_qf', [x])          

    def step(self, x):
        return self._function('step', [x])          

    def StudentT(self, nu, mu, sigma):
        return self._function('student_t', [nu, mu, sigma])          

    def student_t_cdf(self, y, nu, mu, sigma):
        return self._function('student_t_cdf', [y, nu, mu, sigma])          

    def student_t_lccdf(self, y, nu, mu, sigma):
        return self._function('student_t_lccdf', [y, nu, mu, sigma])          

    def student_t_lcdf(self, y, nu, mu, sigma):
        return self._function('student_t_lcdf', [y, nu, mu, sigma])          

    def student_t_lpdf(self, y, nu, mu, sigma):
        return self._function('student_t_lpdf', [y, nu, mu, sigma])          

    def student_t_lupdf(self, y, nu, mu, sigma):
        return self._function('student_t_lupdf', [y, nu, mu, sigma])          

    def student_t_rng(self, nu, mu, sigma):
        return self._function('student_t_rng', [nu, mu, sigma])          

    def sub_col(self, x, i, j, n_rows):
        return self._function('sub_col', [x, i, j, n_rows])          

    def sub_row(self, x, i, j, n_cols):
        return self._function('sub_row', [x, i, j, n_cols])          

    def sum(self, x):
        return self._function('sum', [x])          

    def svd(self, A):
        return self._function('svd', [A])          

    def svd_U(self, A):
        return self._function('svd_U', [A])          

    def svd_V(self, A):
        return self._function('svd_V', [A])          

    def symmetrize_from_lower_tri(self, A):
        return self._function('symmetrize_from_lower_tri', [A])          

    def tail(self, v, n):
        return self._function('tail', [v, n])          

    def tan(self, z):
        return self._function('tan', [z])          

    def tanh(self, z):
        return self._function('tanh', [z])          

    def tcrossprod(self, x):
        return self._function('tcrossprod', [x])          

    def tgamma(self, x):
        return self._function('tgamma', [x])          

    def to_array_1d(self, v):
        return self._function('to_array_1d', [v])          

    def to_array_2d(self, m):
        return self._function('to_array_2d', [m])          

    def to_complex(self, re):
        return self._function('to_complex', [re])          

    def to_int(self, x):
        return self._function('to_int', [x])          

    def to_matrix(self, m):
        return self._function('to_matrix', [m])          

    def to_row_vector(self, m):
        return self._function('to_row_vector', [m])          

    def to_vector(self, m):
        return self._function('to_vector', [m])          

    def trace(self, A):
        return self._function('trace', [A])          

    def trace_gen_quad_form(self, D, A, B):
        return self._function('trace_gen_quad_form', [D, A, B])          

    def trace_quad_form(self, A, B):
        return self._function('trace_quad_form', [A, B])          

    def trigamma(self, x):
        return self._function('trigamma', [x])          

    def trunc(self, x):
        return self._function('trunc', [x])          

    def Uniform(self, alpha, beta):
        return self._function('uniform', [alpha, beta])          

    def uniform_cdf(self, y, alpha, beta):
        return self._function('uniform_cdf', [y, alpha, beta])          

    def uniform_lccdf(self, y, alpha, beta):
        return self._function('uniform_lccdf', [y, alpha, beta])          

    def uniform_lcdf(self, y, alpha, beta):
        return self._function('uniform_lcdf', [y, alpha, beta])          

    def uniform_lpdf(self, y, alpha, beta):
        return self._function('uniform_lpdf', [y, alpha, beta])          

    def uniform_lupdf(self, y, alpha, beta):
        return self._function('uniform_lupdf', [y, alpha, beta])          

    def uniform_rng(self, alpha, beta):
        return self._function('uniform_rng', [alpha, beta])          

    def uniform_simplex(self, n):
        return self._function('uniform_simplex', [n])          

    def variance(self, x):
        return self._function('variance', [x])          

    def VonMises(self, mu, kappa):
        return self._function('von_mises', [mu, kappa])          

    def von_mises_cdf(self, y, mu, kappa):
        return self._function('von_mises_cdf', [y, mu, kappa])          

    def von_mises_lccdf(self, y, mu, kappa):
        return self._function('von_mises_lccdf', [y, mu, kappa])          

    def von_mises_lcdf(self, y, mu, kappa):
        return self._function('von_mises_lcdf', [y, mu, kappa])          

    def von_mises_lpdf(self, y, mu, kappa):
        return self._function('von_mises_lpdf', [y, mu, kappa])          

    def von_mises_lupdf(self, y, mu, kappa):
        return self._function('von_mises_lupdf', [y, mu, kappa])          

    def von_mises_rng(self, mu, kappa):
        return self._function('von_mises_rng', [mu, kappa])          

    def Weibull(self, alpha, sigma):
        return self._function('weibull', [alpha, sigma])          

    def weibull_cdf(self, y, alpha, sigma):
        return self._function('weibull_cdf', [y, alpha, sigma])          

    def weibull_lccdf(self, y, alpha, sigma):
        return self._function('weibull_lccdf', [y, alpha, sigma])          

    def weibull_lcdf(self, y, alpha, sigma):
        return self._function('weibull_lcdf', [y, alpha, sigma])          

    def weibull_lpdf(self, y, alpha, sigma):
        return self._function('weibull_lpdf', [y, alpha, sigma])          

    def weibull_lupdf(self, y, alpha, sigma):
        return self._function('weibull_lupdf', [y, alpha, sigma])          

    def weibull_rng(self, alpha, sigma):
        return self._function('weibull_rng', [alpha, sigma])          

    def Wiener(self, alpha, tau, beta, delta):
        return self._function('wiener', [alpha, tau, beta, delta])          

    def wiener_lpdf(self, y, alpha, tau, beta, delta):
        return self._function('wiener_lpdf', [y, alpha, tau, beta, delta])          

    def wiener_lupdf(self, y, alpha, tau, beta, delta):
        return self._function('wiener_lupdf', [y, alpha, tau, beta, delta])          

    def Wishart(self, nu, Sigma):
        return self._function('wishart', [nu, Sigma])          

    def WishartCholesky(self, nu, L_S):
        return self._function('wishart_cholesky', [nu, L_S])          

    def wishart_cholesky_lpdf(self, L_W, nu, L_S):
        return self._function('wishart_cholesky_lpdf', [L_W, nu, L_S])          

    def wishart_cholesky_lupdf(self, L_W, nu, L_S):
        return self._function('wishart_cholesky_lupdf', [L_W, nu, L_S])          

    def wishart_cholesky_rng(self, nu, L_S):
        return self._function('wishart_cholesky_rng', [nu, L_S])          

    def wishart_lpdf(self, W, nu, Sigma):
        return self._function('wishart_lpdf', [W, nu, Sigma])          

    def wishart_lupdf(self, W, nu, Sigma):
        return self._function('wishart_lupdf', [W, nu, Sigma])          

    def wishart_rng(self, nu, Sigma):
        return self._function('wishart_rng', [nu, Sigma])          

    def zeros_array(self, n):
        return self._function('zeros_array', [n])          

    def zeros_int_array(self, n):
        return self._function('zeros_int_array', [n])          

    def zeros_row_vector(self, n):
        return self._function('zeros_row_vector', [n])          
