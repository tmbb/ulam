
from py_expr import PyExpr, python_to_ast
from stan_ast import FunctionCall

class StanFunctionsLibrary:
    def _function(self, function_name, args):
        arguments_as_ast = [python_to_ast(arg) for arg in args]
        function_call_ast = FunctionCall(function_name, arguments_as_ast)
        self.functions[function_name] = self.functions.get(function_name, 0) + 1

        return PyExpr(self, function_call_ast)
              

    def abs(self, z):
        self._function('abs', [z])          

    def acos(self, z):
        self._function('acos', [z])          

    def acosh(self, z):
        self._function('acosh', [z])          

    def add_diag(self, m, d):
        self._function('add_diag', [m, d])          

    def algebra_solver(self, algebra_system, y_guess, theta, x_r, x_i, rel_tol, f_tol, max_steps):
        self._function('algebra_solver', [algebra_system, y_guess, theta, x_r, x_i, rel_tol, f_tol, max_steps])          

    def algebra_solver_newton(self, algebra_system, y_guess, theta, x_r, x_i):
        self._function('algebra_solver_newton', [algebra_system, y_guess, theta, x_r, x_i])          

    def append_array(self, x, y):
        self._function('append_array', [x, y])          

    def append_col(self, x, y):
        self._function('append_col', [x, y])          

    def append_row(self, x, y):
        self._function('append_row', [x, y])          

    def arg(self, z):
        self._function('arg', [z])          

    def asin(self, z):
        self._function('asin', [z])          

    def asinh(self, z):
        self._function('asinh', [z])          

    def atan(self, z):
        self._function('atan', [z])          

    def atan2(self, y, x):
        self._function('atan2', [y, x])          

    def atanh(self, z):
        self._function('atanh', [z])          

    def bernoulli(self, theta):
        self._function('bernoulli', [theta])          

    def bernoulli_cdf(self, y, theta):
        self._function('bernoulli_cdf', [y, theta])          

    def bernoulli_lccdf(self, y, theta):
        self._function('bernoulli_lccdf', [y, theta])          

    def bernoulli_lcdf(self, y, theta):
        self._function('bernoulli_lcdf', [y, theta])          

    def bernoulli_logit(self, alpha):
        self._function('bernoulli_logit', [alpha])          

    def bernoulli_logit_glm(self, x, alpha, beta):
        self._function('bernoulli_logit_glm', [x, alpha, beta])          

    def bernoulli_logit_glm_lpmf(self, y, x, alpha, beta):
        self._function('bernoulli_logit_glm_lpmf', [y, x, alpha, beta])          

    def bernoulli_logit_glm_lupmf(self, y, x, alpha, beta):
        self._function('bernoulli_logit_glm_lupmf', [y, x, alpha, beta])          

    def bernoulli_logit_glm_rng(self, x, alpha, beta):
        self._function('bernoulli_logit_glm_rng', [x, alpha, beta])          

    def bernoulli_logit_glm_with_left_and_right_censoring(self, x, alpha, beta, event_left, event_right):
        self._function('bernoulli_logit_glm_with_left_and_right_censoring', [x, alpha, beta, event_left, event_right])          

    def bernoulli_logit_glm_with_left_and_right_censoring_lpmf(self, y, x, alpha, beta, event_left, event_right):
        self._function('bernoulli_logit_glm_with_left_and_right_censoring_lpmf', [y, x, alpha, beta, event_left, event_right])          

    def bernoulli_logit_glm_with_left_and_right_censoring_rng(self, x, alpha, beta, event_left, event_right):
        self._function('bernoulli_logit_glm_with_left_and_right_censoring_rng', [x, alpha, beta, event_left, event_right])          

    def bernoulli_logit_glm_with_left_censoring(self, x, alpha, beta, event):
        self._function('bernoulli_logit_glm_with_left_censoring', [x, alpha, beta, event])          

    def bernoulli_logit_glm_with_left_censoring_lpmf(self, y, x, alpha, beta, event):
        self._function('bernoulli_logit_glm_with_left_censoring_lpmf', [y, x, alpha, beta, event])          

    def bernoulli_logit_glm_with_left_censoring_rng(self, y, x, alpha, beta, event):
        self._function('bernoulli_logit_glm_with_left_censoring_rng', [y, x, alpha, beta, event])          

    def bernoulli_logit_glm_with_right_censoring(self, x, alpha, beta):
        self._function('bernoulli_logit_glm_with_right_censoring', [x, alpha, beta])          

    def bernoulli_logit_glm_with_right_censoring_lpmf(self, y, x, alpha, beta, event):
        self._function('bernoulli_logit_glm_with_right_censoring_lpmf', [y, x, alpha, beta, event])          

    def bernoulli_logit_glm_with_right_censoring_rng(self, y, x, alpha, beta, event):
        self._function('bernoulli_logit_glm_with_right_censoring_rng', [y, x, alpha, beta, event])          

    def bernoulli_logit_lpmf(self, y, alpha):
        self._function('bernoulli_logit_lpmf', [y, alpha])          

    def bernoulli_logit_lupmf(self, y, alpha):
        self._function('bernoulli_logit_lupmf', [y, alpha])          

    def bernoulli_logit_rng(self, alpha):
        self._function('bernoulli_logit_rng', [alpha])          

    def bernoulli_logit_with_left_and_right_censoring(self, alpha, event_left, event_right):
        self._function('bernoulli_logit_with_left_and_right_censoring', [alpha, event_left, event_right])          

    def bernoulli_logit_with_left_and_right_censoring_lpmf(self, y, alpha, event_left, event_right):
        self._function('bernoulli_logit_with_left_and_right_censoring_lpmf', [y, alpha, event_left, event_right])          

    def bernoulli_logit_with_left_and_right_censoring_rng(self, alpha, event_left, event_right):
        self._function('bernoulli_logit_with_left_and_right_censoring_rng', [alpha, event_left, event_right])          

    def bernoulli_logit_with_left_censoring(self, alpha, event):
        self._function('bernoulli_logit_with_left_censoring', [alpha, event])          

    def bernoulli_logit_with_left_censoring_lpmf(self, y, alpha, event):
        self._function('bernoulli_logit_with_left_censoring_lpmf', [y, alpha, event])          

    def bernoulli_logit_with_left_censoring_rng(self, y, alpha, event):
        self._function('bernoulli_logit_with_left_censoring_rng', [y, alpha, event])          

    def bernoulli_logit_with_right_censoring(self, alpha):
        self._function('bernoulli_logit_with_right_censoring', [alpha])          

    def bernoulli_logit_with_right_censoring_lpmf(self, y, alpha, event):
        self._function('bernoulli_logit_with_right_censoring_lpmf', [y, alpha, event])          

    def bernoulli_logit_with_right_censoring_rng(self, y, alpha, event):
        self._function('bernoulli_logit_with_right_censoring_rng', [y, alpha, event])          

    def bernoulli_lpmf(self, y, theta):
        self._function('bernoulli_lpmf', [y, theta])          

    def bernoulli_lupmf(self, y, theta):
        self._function('bernoulli_lupmf', [y, theta])          

    def bernoulli_rng(self, theta):
        self._function('bernoulli_rng', [theta])          

    def bernoulli_with_left_and_right_censoring(self, theta, event_left, event_right):
        self._function('bernoulli_with_left_and_right_censoring', [theta, event_left, event_right])          

    def bernoulli_with_left_and_right_censoring_lpmf(self, y, theta, event_left, event_right):
        self._function('bernoulli_with_left_and_right_censoring_lpmf', [y, theta, event_left, event_right])          

    def bernoulli_with_left_and_right_censoring_rng(self, theta, event_left, event_right):
        self._function('bernoulli_with_left_and_right_censoring_rng', [theta, event_left, event_right])          

    def bernoulli_with_left_censoring(self, theta, event):
        self._function('bernoulli_with_left_censoring', [theta, event])          

    def bernoulli_with_left_censoring_lpmf(self, y, theta, event):
        self._function('bernoulli_with_left_censoring_lpmf', [y, theta, event])          

    def bernoulli_with_left_censoring_rng(self, y, theta, event):
        self._function('bernoulli_with_left_censoring_rng', [y, theta, event])          

    def bernoulli_with_right_censoring(self, theta):
        self._function('bernoulli_with_right_censoring', [theta])          

    def bernoulli_with_right_censoring_lpmf(self, y, theta, event):
        self._function('bernoulli_with_right_censoring_lpmf', [y, theta, event])          

    def bernoulli_with_right_censoring_rng(self, y, theta, event):
        self._function('bernoulli_with_right_censoring_rng', [y, theta, event])          

    def bessel_first_kind(self, v, x):
        self._function('bessel_first_kind', [v, x])          

    def bessel_second_kind(self, v, x):
        self._function('bessel_second_kind', [v, x])          

    def beta(self, alpha, beta):
        self._function('beta', [alpha, beta])          

    def beta_binomial(self, N, alpha, beta):
        self._function('beta_binomial', [N, alpha, beta])          

    def beta_binomial_cdf(self, n, N, alpha, beta):
        self._function('beta_binomial_cdf', [n, N, alpha, beta])          

    def beta_binomial_lccdf(self, n, N, alpha, beta):
        self._function('beta_binomial_lccdf', [n, N, alpha, beta])          

    def beta_binomial_lcdf(self, n, N, alpha, beta):
        self._function('beta_binomial_lcdf', [n, N, alpha, beta])          

    def beta_binomial_lpmf(self, n, N, alpha, beta):
        self._function('beta_binomial_lpmf', [n, N, alpha, beta])          

    def beta_binomial_lupmf(self, n, N, alpha, beta):
        self._function('beta_binomial_lupmf', [n, N, alpha, beta])          

    def beta_binomial_rng(self, N, alpha, beta):
        self._function('beta_binomial_rng', [N, alpha, beta])          

    def beta_binomial_with_left_and_right_censoring(self, N, alpha, beta, event_left, event_right):
        self._function('beta_binomial_with_left_and_right_censoring', [N, alpha, beta, event_left, event_right])          

    def beta_binomial_with_left_and_right_censoring_lpmf(self, n, N, alpha, beta, event_left, event_right):
        self._function('beta_binomial_with_left_and_right_censoring_lpmf', [n, N, alpha, beta, event_left, event_right])          

    def beta_binomial_with_left_and_right_censoring_rng(self, N, alpha, beta, event_left, event_right):
        self._function('beta_binomial_with_left_and_right_censoring_rng', [N, alpha, beta, event_left, event_right])          

    def beta_binomial_with_left_censoring(self, N, alpha, beta, event):
        self._function('beta_binomial_with_left_censoring', [N, alpha, beta, event])          

    def beta_binomial_with_left_censoring_lpmf(self, n, N, alpha, beta, event):
        self._function('beta_binomial_with_left_censoring_lpmf', [n, N, alpha, beta, event])          

    def beta_binomial_with_left_censoring_rng(self, n, N, alpha, beta, event):
        self._function('beta_binomial_with_left_censoring_rng', [n, N, alpha, beta, event])          

    def beta_binomial_with_right_censoring(self, N, alpha, beta):
        self._function('beta_binomial_with_right_censoring', [N, alpha, beta])          

    def beta_binomial_with_right_censoring_lpmf(self, n, N, alpha, beta, event):
        self._function('beta_binomial_with_right_censoring_lpmf', [n, N, alpha, beta, event])          

    def beta_binomial_with_right_censoring_rng(self, n, N, alpha, beta, event):
        self._function('beta_binomial_with_right_censoring_rng', [n, N, alpha, beta, event])          

    def beta_cdf(self, theta, alpha, beta):
        self._function('beta_cdf', [theta, alpha, beta])          

    def beta_lccdf(self, theta, alpha, beta):
        self._function('beta_lccdf', [theta, alpha, beta])          

    def beta_lcdf(self, theta, alpha, beta):
        self._function('beta_lcdf', [theta, alpha, beta])          

    def beta_lpdf(self, theta, alpha, beta):
        self._function('beta_lpdf', [theta, alpha, beta])          

    def beta_lupdf(self, theta, alpha, beta):
        self._function('beta_lupdf', [theta, alpha, beta])          

    def beta_proportion_lccdf(self, theta, mu, kappa):
        self._function('beta_proportion_lccdf', [theta, mu, kappa])          

    def beta_proportion_lcdf(self, theta, mu, kappa):
        self._function('beta_proportion_lcdf', [theta, mu, kappa])          

    def beta_proportion_rng(self, mu, kappa):
        self._function('beta_proportion_rng', [mu, kappa])          

    def beta_rng(self, alpha, beta):
        self._function('beta_rng', [alpha, beta])          

    def beta_with_left_and_right_censoring(self, alpha, beta, event_left, event_right):
        self._function('beta_with_left_and_right_censoring', [alpha, beta, event_left, event_right])          

    def beta_with_left_and_right_censoring_lpdf(self, theta, alpha, beta, event_left, event_right):
        self._function('beta_with_left_and_right_censoring_lpdf', [theta, alpha, beta, event_left, event_right])          

    def beta_with_left_and_right_censoring_rng(self, alpha, beta, event_left, event_right):
        self._function('beta_with_left_and_right_censoring_rng', [alpha, beta, event_left, event_right])          

    def beta_with_left_censoring(self, alpha, beta, event):
        self._function('beta_with_left_censoring', [alpha, beta, event])          

    def beta_with_left_censoring_lpdf(self, theta, alpha, beta, event):
        self._function('beta_with_left_censoring_lpdf', [theta, alpha, beta, event])          

    def beta_with_left_censoring_rng(self, theta, alpha, beta, event):
        self._function('beta_with_left_censoring_rng', [theta, alpha, beta, event])          

    def beta_with_right_censoring(self, alpha, beta):
        self._function('beta_with_right_censoring', [alpha, beta])          

    def beta_with_right_censoring_lpdf(self, theta, alpha, beta, event):
        self._function('beta_with_right_censoring_lpdf', [theta, alpha, beta, event])          

    def beta_with_right_censoring_rng(self, theta, alpha, beta, event):
        self._function('beta_with_right_censoring_rng', [theta, alpha, beta, event])          

    def binary_log_loss(self, y, y_hat):
        self._function('binary_log_loss', [y, y_hat])          

    def binomial(self, N, theta):
        self._function('binomial', [N, theta])          

    def binomial_cdf(self, n, N, theta):
        self._function('binomial_cdf', [n, N, theta])          

    def binomial_lccdf(self, n, N, theta):
        self._function('binomial_lccdf', [n, N, theta])          

    def binomial_lcdf(self, n, N, theta):
        self._function('binomial_lcdf', [n, N, theta])          

    def binomial_logit(self, N, alpha):
        self._function('binomial_logit', [N, alpha])          

    def binomial_logit_lpmf(self, n, N, alpha):
        self._function('binomial_logit_lpmf', [n, N, alpha])          

    def binomial_logit_lupmf(self, n, N, alpha):
        self._function('binomial_logit_lupmf', [n, N, alpha])          

    def binomial_logit_with_left_and_right_censoring(self, N, alpha, event_left, event_right):
        self._function('binomial_logit_with_left_and_right_censoring', [N, alpha, event_left, event_right])          

    def binomial_logit_with_left_and_right_censoring_lpmf(self, n, N, alpha, event_left, event_right):
        self._function('binomial_logit_with_left_and_right_censoring_lpmf', [n, N, alpha, event_left, event_right])          

    def binomial_logit_with_left_and_right_censoring_rng(self, N, alpha, event_left, event_right):
        self._function('binomial_logit_with_left_and_right_censoring_rng', [N, alpha, event_left, event_right])          

    def binomial_logit_with_left_censoring(self, N, alpha, event):
        self._function('binomial_logit_with_left_censoring', [N, alpha, event])          

    def binomial_logit_with_left_censoring_lpmf(self, n, N, alpha, event):
        self._function('binomial_logit_with_left_censoring_lpmf', [n, N, alpha, event])          

    def binomial_logit_with_left_censoring_rng(self, n, N, alpha, event):
        self._function('binomial_logit_with_left_censoring_rng', [n, N, alpha, event])          

    def binomial_logit_with_right_censoring(self, N, alpha):
        self._function('binomial_logit_with_right_censoring', [N, alpha])          

    def binomial_logit_with_right_censoring_lpmf(self, n, N, alpha, event):
        self._function('binomial_logit_with_right_censoring_lpmf', [n, N, alpha, event])          

    def binomial_logit_with_right_censoring_rng(self, n, N, alpha, event):
        self._function('binomial_logit_with_right_censoring_rng', [n, N, alpha, event])          

    def binomial_lpmf(self, n, N, theta):
        self._function('binomial_lpmf', [n, N, theta])          

    def binomial_lupmf(self, n, N, theta):
        self._function('binomial_lupmf', [n, N, theta])          

    def binomial_rng(self, N, theta):
        self._function('binomial_rng', [N, theta])          

    def binomial_with_left_and_right_censoring(self, N, theta, event_left, event_right):
        self._function('binomial_with_left_and_right_censoring', [N, theta, event_left, event_right])          

    def binomial_with_left_and_right_censoring_lpmf(self, n, N, theta, event_left, event_right):
        self._function('binomial_with_left_and_right_censoring_lpmf', [n, N, theta, event_left, event_right])          

    def binomial_with_left_and_right_censoring_rng(self, N, theta, event_left, event_right):
        self._function('binomial_with_left_and_right_censoring_rng', [N, theta, event_left, event_right])          

    def binomial_with_left_censoring(self, N, theta, event):
        self._function('binomial_with_left_censoring', [N, theta, event])          

    def binomial_with_left_censoring_lpmf(self, n, N, theta, event):
        self._function('binomial_with_left_censoring_lpmf', [n, N, theta, event])          

    def binomial_with_left_censoring_rng(self, n, N, theta, event):
        self._function('binomial_with_left_censoring_rng', [n, N, theta, event])          

    def binomial_with_right_censoring(self, N, theta):
        self._function('binomial_with_right_censoring', [N, theta])          

    def binomial_with_right_censoring_lpmf(self, n, N, theta, event):
        self._function('binomial_with_right_censoring_lpmf', [n, N, theta, event])          

    def binomial_with_right_censoring_rng(self, n, N, theta, event):
        self._function('binomial_with_right_censoring_rng', [n, N, theta, event])          

    def block(self, x, i, j, n_rows, n_cols):
        self._function('block', [x, i, j, n_rows, n_cols])          

    def categorical(self, theta):
        self._function('categorical', [theta])          

    def categorical_logit(self, beta):
        self._function('categorical_logit', [beta])          

    def categorical_logit_glm(self, x, alpha, beta):
        self._function('categorical_logit_glm', [x, alpha, beta])          

    def categorical_logit_glm_lpmf(self, y, x, alpha, beta):
        self._function('categorical_logit_glm_lpmf', [y, x, alpha, beta])          

    def categorical_logit_glm_lupmf(self, y, x, alpha, beta):
        self._function('categorical_logit_glm_lupmf', [y, x, alpha, beta])          

    def categorical_logit_glm_with_left_and_right_censoring(self, x, alpha, beta, event_left, event_right):
        self._function('categorical_logit_glm_with_left_and_right_censoring', [x, alpha, beta, event_left, event_right])          

    def categorical_logit_glm_with_left_and_right_censoring_lpmf(self, y, x, alpha, beta, event_left, event_right):
        self._function('categorical_logit_glm_with_left_and_right_censoring_lpmf', [y, x, alpha, beta, event_left, event_right])          

    def categorical_logit_glm_with_left_and_right_censoring_rng(self, x, alpha, beta, event_left, event_right):
        self._function('categorical_logit_glm_with_left_and_right_censoring_rng', [x, alpha, beta, event_left, event_right])          

    def categorical_logit_glm_with_left_censoring(self, x, alpha, beta, event):
        self._function('categorical_logit_glm_with_left_censoring', [x, alpha, beta, event])          

    def categorical_logit_glm_with_left_censoring_lpmf(self, y, x, alpha, beta, event):
        self._function('categorical_logit_glm_with_left_censoring_lpmf', [y, x, alpha, beta, event])          

    def categorical_logit_glm_with_left_censoring_rng(self, y, x, alpha, beta, event):
        self._function('categorical_logit_glm_with_left_censoring_rng', [y, x, alpha, beta, event])          

    def categorical_logit_glm_with_right_censoring(self, x, alpha, beta):
        self._function('categorical_logit_glm_with_right_censoring', [x, alpha, beta])          

    def categorical_logit_glm_with_right_censoring_lpmf(self, y, x, alpha, beta, event):
        self._function('categorical_logit_glm_with_right_censoring_lpmf', [y, x, alpha, beta, event])          

    def categorical_logit_glm_with_right_censoring_rng(self, y, x, alpha, beta, event):
        self._function('categorical_logit_glm_with_right_censoring_rng', [y, x, alpha, beta, event])          

    def categorical_logit_lpmf(self, y, beta):
        self._function('categorical_logit_lpmf', [y, beta])          

    def categorical_logit_lupmf(self, y, beta):
        self._function('categorical_logit_lupmf', [y, beta])          

    def categorical_logit_rng(self, beta):
        self._function('categorical_logit_rng', [beta])          

    def categorical_logit_with_left_and_right_censoring(self, beta, event_left, event_right):
        self._function('categorical_logit_with_left_and_right_censoring', [beta, event_left, event_right])          

    def categorical_logit_with_left_and_right_censoring_lpmf(self, y, beta, event_left, event_right):
        self._function('categorical_logit_with_left_and_right_censoring_lpmf', [y, beta, event_left, event_right])          

    def categorical_logit_with_left_and_right_censoring_rng(self, beta, event_left, event_right):
        self._function('categorical_logit_with_left_and_right_censoring_rng', [beta, event_left, event_right])          

    def categorical_logit_with_left_censoring(self, beta, event):
        self._function('categorical_logit_with_left_censoring', [beta, event])          

    def categorical_logit_with_left_censoring_lpmf(self, y, beta, event):
        self._function('categorical_logit_with_left_censoring_lpmf', [y, beta, event])          

    def categorical_logit_with_left_censoring_rng(self, y, beta, event):
        self._function('categorical_logit_with_left_censoring_rng', [y, beta, event])          

    def categorical_logit_with_right_censoring(self, beta):
        self._function('categorical_logit_with_right_censoring', [beta])          

    def categorical_logit_with_right_censoring_lpmf(self, y, beta, event):
        self._function('categorical_logit_with_right_censoring_lpmf', [y, beta, event])          

    def categorical_logit_with_right_censoring_rng(self, y, beta, event):
        self._function('categorical_logit_with_right_censoring_rng', [y, beta, event])          

    def categorical_lpmf(self, y, theta):
        self._function('categorical_lpmf', [y, theta])          

    def categorical_lupmf(self, y, theta):
        self._function('categorical_lupmf', [y, theta])          

    def categorical_rng(self, theta):
        self._function('categorical_rng', [theta])          

    def categorical_with_left_and_right_censoring(self, theta, event_left, event_right):
        self._function('categorical_with_left_and_right_censoring', [theta, event_left, event_right])          

    def categorical_with_left_and_right_censoring_lpmf(self, y, theta, event_left, event_right):
        self._function('categorical_with_left_and_right_censoring_lpmf', [y, theta, event_left, event_right])          

    def categorical_with_left_and_right_censoring_rng(self, theta, event_left, event_right):
        self._function('categorical_with_left_and_right_censoring_rng', [theta, event_left, event_right])          

    def categorical_with_left_censoring(self, theta, event):
        self._function('categorical_with_left_censoring', [theta, event])          

    def categorical_with_left_censoring_lpmf(self, y, theta, event):
        self._function('categorical_with_left_censoring_lpmf', [y, theta, event])          

    def categorical_with_left_censoring_rng(self, y, theta, event):
        self._function('categorical_with_left_censoring_rng', [y, theta, event])          

    def categorical_with_right_censoring(self, theta):
        self._function('categorical_with_right_censoring', [theta])          

    def categorical_with_right_censoring_lpmf(self, y, theta, event):
        self._function('categorical_with_right_censoring_lpmf', [y, theta, event])          

    def categorical_with_right_censoring_rng(self, y, theta, event):
        self._function('categorical_with_right_censoring_rng', [y, theta, event])          

    def cauchy(self, mu, sigma):
        self._function('cauchy', [mu, sigma])          

    def cauchy_cdf(self, y, mu, sigma):
        self._function('cauchy_cdf', [y, mu, sigma])          

    def cauchy_lccdf(self, y, mu, sigma):
        self._function('cauchy_lccdf', [y, mu, sigma])          

    def cauchy_lcdf(self, y, mu, sigma):
        self._function('cauchy_lcdf', [y, mu, sigma])          

    def cauchy_lpdf(self, y, mu, sigma):
        self._function('cauchy_lpdf', [y, mu, sigma])          

    def cauchy_lupdf(self, y, mu, sigma):
        self._function('cauchy_lupdf', [y, mu, sigma])          

    def cauchy_rng(self, mu, sigma):
        self._function('cauchy_rng', [mu, sigma])          

    def cauchy_with_left_and_right_censoring(self, mu, sigma, event_left, event_right):
        self._function('cauchy_with_left_and_right_censoring', [mu, sigma, event_left, event_right])          

    def cauchy_with_left_and_right_censoring_lpdf(self, y, mu, sigma, event_left, event_right):
        self._function('cauchy_with_left_and_right_censoring_lpdf', [y, mu, sigma, event_left, event_right])          

    def cauchy_with_left_and_right_censoring_rng(self, mu, sigma, event_left, event_right):
        self._function('cauchy_with_left_and_right_censoring_rng', [mu, sigma, event_left, event_right])          

    def cauchy_with_left_censoring(self, mu, sigma, event):
        self._function('cauchy_with_left_censoring', [mu, sigma, event])          

    def cauchy_with_left_censoring_lpdf(self, y, mu, sigma, event):
        self._function('cauchy_with_left_censoring_lpdf', [y, mu, sigma, event])          

    def cauchy_with_left_censoring_rng(self, y, mu, sigma, event):
        self._function('cauchy_with_left_censoring_rng', [y, mu, sigma, event])          

    def cauchy_with_right_censoring(self, mu, sigma):
        self._function('cauchy_with_right_censoring', [mu, sigma])          

    def cauchy_with_right_censoring_lpdf(self, y, mu, sigma, event):
        self._function('cauchy_with_right_censoring_lpdf', [y, mu, sigma, event])          

    def cauchy_with_right_censoring_rng(self, y, mu, sigma, event):
        self._function('cauchy_with_right_censoring_rng', [y, mu, sigma, event])          

    def cbrt(self, x):
        self._function('cbrt', [x])          

    def ceil(self, x):
        self._function('ceil', [x])          

    def chi_square(self, nu):
        self._function('chi_square', [nu])          

    def chi_square_cdf(self, y, nu):
        self._function('chi_square_cdf', [y, nu])          

    def chi_square_lccdf(self, y, nu):
        self._function('chi_square_lccdf', [y, nu])          

    def chi_square_lcdf(self, y, nu):
        self._function('chi_square_lcdf', [y, nu])          

    def chi_square_lpdf(self, y, nu):
        self._function('chi_square_lpdf', [y, nu])          

    def chi_square_lupdf(self, y, nu):
        self._function('chi_square_lupdf', [y, nu])          

    def chi_square_rng(self, nu):
        self._function('chi_square_rng', [nu])          

    def chi_square_with_left_and_right_censoring(self, nu, event_left, event_right):
        self._function('chi_square_with_left_and_right_censoring', [nu, event_left, event_right])          

    def chi_square_with_left_and_right_censoring_lpdf(self, y, nu, event_left, event_right):
        self._function('chi_square_with_left_and_right_censoring_lpdf', [y, nu, event_left, event_right])          

    def chi_square_with_left_and_right_censoring_rng(self, nu, event_left, event_right):
        self._function('chi_square_with_left_and_right_censoring_rng', [nu, event_left, event_right])          

    def chi_square_with_left_censoring(self, nu, event):
        self._function('chi_square_with_left_censoring', [nu, event])          

    def chi_square_with_left_censoring_lpdf(self, y, nu, event):
        self._function('chi_square_with_left_censoring_lpdf', [y, nu, event])          

    def chi_square_with_left_censoring_rng(self, y, nu, event):
        self._function('chi_square_with_left_censoring_rng', [y, nu, event])          

    def chi_square_with_right_censoring(self, nu):
        self._function('chi_square_with_right_censoring', [nu])          

    def chi_square_with_right_censoring_lpdf(self, y, nu, event):
        self._function('chi_square_with_right_censoring_lpdf', [y, nu, event])          

    def chi_square_with_right_censoring_rng(self, y, nu, event):
        self._function('chi_square_with_right_censoring_rng', [y, nu, event])          

    def chol2inv(self, L):
        self._function('chol2inv', [L])          

    def cholesky_decompose(self, A):
        self._function('cholesky_decompose', [A])          

    def choose(self, x, y):
        self._function('choose', [x, y])          

    def col(self, x, n):
        self._function('col', [x, n])          

    def cols(self, x):
        self._function('cols', [x])          

    def columns_dot_product(self, x, y):
        self._function('columns_dot_product', [x, y])          

    def columns_dot_self(self, x):
        self._function('columns_dot_self', [x])          

    def complex_schur_decompose(self, A):
        self._function('complex_schur_decompose', [A])          

    def complex_schur_decompose_t(self, A):
        self._function('complex_schur_decompose_t', [A])          

    def complex_schur_decompose_u(self, A):
        self._function('complex_schur_decompose_u', [A])          

    def conj(self, z):
        self._function('conj', [z])          

    def cos(self, z):
        self._function('cos', [z])          

    def cosh(self, z):
        self._function('cosh', [z])          

    def cov_exp_quad(self, x, alpha, rho):
        self._function('cov_exp_quad', [x, alpha, rho])          

    def crossprod(self, x):
        self._function('crossprod', [x])          

    def csr_extract(self, a):
        self._function('csr_extract', [a])          

    def csr_extract_u(self, a):
        self._function('csr_extract_u', [a])          

    def csr_extract_v(self, a):
        self._function('csr_extract_v', [a])          

    def csr_extract_w(self, a):
        self._function('csr_extract_w', [a])          

    def csr_matrix_times_vector(self, m, n, w, v, u, b):
        self._function('csr_matrix_times_vector', [m, n, w, v, u, b])          

    def csr_to_dense_matrix(self, m, n, w, v, u):
        self._function('csr_to_dense_matrix', [m, n, w, v, u])          

    def cumulative_sum(self, x):
        self._function('cumulative_sum', [x])          

    def determinant(self, A):
        self._function('determinant', [A])          

    def diag_matrix(self, x):
        self._function('diag_matrix', [x])          

    def diag_post_multiply(self, m, v):
        self._function('diag_post_multiply', [m, v])          

    def diag_pre_multiply(self, v, m):
        self._function('diag_pre_multiply', [v, m])          

    def diagonal(self, x):
        self._function('diagonal', [x])          

    def digamma(self, x):
        self._function('digamma', [x])          

    def dims(self, x):
        self._function('dims', [x])          

    def dirichlet(self, alpha):
        self._function('dirichlet', [alpha])          

    def dirichlet_lpdf(self, theta, alpha):
        self._function('dirichlet_lpdf', [theta, alpha])          

    def dirichlet_lupdf(self, theta, alpha):
        self._function('dirichlet_lupdf', [theta, alpha])          

    def dirichlet_rng(self, alpha):
        self._function('dirichlet_rng', [alpha])          

    def dirichlet_with_left_and_right_censoring(self, alpha, event_left, event_right):
        self._function('dirichlet_with_left_and_right_censoring', [alpha, event_left, event_right])          

    def dirichlet_with_left_and_right_censoring_lpdf(self, theta, alpha, event_left, event_right):
        self._function('dirichlet_with_left_and_right_censoring_lpdf', [theta, alpha, event_left, event_right])          

    def dirichlet_with_left_and_right_censoring_rng(self, alpha, event_left, event_right):
        self._function('dirichlet_with_left_and_right_censoring_rng', [alpha, event_left, event_right])          

    def dirichlet_with_left_censoring(self, alpha, event):
        self._function('dirichlet_with_left_censoring', [alpha, event])          

    def dirichlet_with_left_censoring_lpdf(self, theta, alpha, event):
        self._function('dirichlet_with_left_censoring_lpdf', [theta, alpha, event])          

    def dirichlet_with_left_censoring_rng(self, theta, alpha, event):
        self._function('dirichlet_with_left_censoring_rng', [theta, alpha, event])          

    def dirichlet_with_right_censoring(self, alpha):
        self._function('dirichlet_with_right_censoring', [alpha])          

    def dirichlet_with_right_censoring_lpdf(self, theta, alpha, event):
        self._function('dirichlet_with_right_censoring_lpdf', [theta, alpha, event])          

    def dirichlet_with_right_censoring_rng(self, theta, alpha, event):
        self._function('dirichlet_with_right_censoring_rng', [theta, alpha, event])          

    def discrete_range(self, l, u):
        self._function('discrete_range', [l, u])          

    def discrete_range_cdf(self, y, l, u):
        self._function('discrete_range_cdf', [y, l, u])          

    def discrete_range_lccdf(self, y, l, u):
        self._function('discrete_range_lccdf', [y, l, u])          

    def discrete_range_lcdf(self, y, l, u):
        self._function('discrete_range_lcdf', [y, l, u])          

    def discrete_range_lpmf(self, y, l, u):
        self._function('discrete_range_lpmf', [y, l, u])          

    def discrete_range_lupmf(self, y, l, u):
        self._function('discrete_range_lupmf', [y, l, u])          

    def discrete_range_rng(self, l, u):
        self._function('discrete_range_rng', [l, u])          

    def discrete_range_with_left_and_right_censoring(self, l, u, event_left, event_right):
        self._function('discrete_range_with_left_and_right_censoring', [l, u, event_left, event_right])          

    def discrete_range_with_left_and_right_censoring_lpmf(self, y, l, u, event_left, event_right):
        self._function('discrete_range_with_left_and_right_censoring_lpmf', [y, l, u, event_left, event_right])          

    def discrete_range_with_left_and_right_censoring_rng(self, l, u, event_left, event_right):
        self._function('discrete_range_with_left_and_right_censoring_rng', [l, u, event_left, event_right])          

    def discrete_range_with_left_censoring(self, l, u, event):
        self._function('discrete_range_with_left_censoring', [l, u, event])          

    def discrete_range_with_left_censoring_lpmf(self, y, l, u, event):
        self._function('discrete_range_with_left_censoring_lpmf', [y, l, u, event])          

    def discrete_range_with_left_censoring_rng(self, y, l, u, event):
        self._function('discrete_range_with_left_censoring_rng', [y, l, u, event])          

    def discrete_range_with_right_censoring(self, l, u):
        self._function('discrete_range_with_right_censoring', [l, u])          

    def discrete_range_with_right_censoring_lpmf(self, y, l, u, event):
        self._function('discrete_range_with_right_censoring_lpmf', [y, l, u, event])          

    def discrete_range_with_right_censoring_rng(self, y, l, u, event):
        self._function('discrete_range_with_right_censoring_rng', [y, l, u, event])          

    def distance(self, x, y):
        self._function('distance', [x, y])          

    def dot_product(self, x, y):
        self._function('dot_product', [x, y])          

    def dot_self(self, x):
        self._function('dot_self', [x])          

    def double_exponential(self, mu, sigma):
        self._function('double_exponential', [mu, sigma])          

    def double_exponential_cdf(self, y, mu, sigma):
        self._function('double_exponential_cdf', [y, mu, sigma])          

    def double_exponential_lccdf(self, y, mu, sigma):
        self._function('double_exponential_lccdf', [y, mu, sigma])          

    def double_exponential_lcdf(self, y, mu, sigma):
        self._function('double_exponential_lcdf', [y, mu, sigma])          

    def double_exponential_lpdf(self, y, mu, sigma):
        self._function('double_exponential_lpdf', [y, mu, sigma])          

    def double_exponential_lupdf(self, y, mu, sigma):
        self._function('double_exponential_lupdf', [y, mu, sigma])          

    def double_exponential_rng(self, mu, sigma):
        self._function('double_exponential_rng', [mu, sigma])          

    def double_exponential_with_left_and_right_censoring(self, mu, sigma, event_left, event_right):
        self._function('double_exponential_with_left_and_right_censoring', [mu, sigma, event_left, event_right])          

    def double_exponential_with_left_and_right_censoring_lpdf(self, y, mu, sigma, event_left, event_right):
        self._function('double_exponential_with_left_and_right_censoring_lpdf', [y, mu, sigma, event_left, event_right])          

    def double_exponential_with_left_and_right_censoring_rng(self, mu, sigma, event_left, event_right):
        self._function('double_exponential_with_left_and_right_censoring_rng', [mu, sigma, event_left, event_right])          

    def double_exponential_with_left_censoring(self, mu, sigma, event):
        self._function('double_exponential_with_left_censoring', [mu, sigma, event])          

    def double_exponential_with_left_censoring_lpdf(self, y, mu, sigma, event):
        self._function('double_exponential_with_left_censoring_lpdf', [y, mu, sigma, event])          

    def double_exponential_with_left_censoring_rng(self, y, mu, sigma, event):
        self._function('double_exponential_with_left_censoring_rng', [y, mu, sigma, event])          

    def double_exponential_with_right_censoring(self, mu, sigma):
        self._function('double_exponential_with_right_censoring', [mu, sigma])          

    def double_exponential_with_right_censoring_lpdf(self, y, mu, sigma, event):
        self._function('double_exponential_with_right_censoring_lpdf', [y, mu, sigma, event])          

    def double_exponential_with_right_censoring_rng(self, y, mu, sigma, event):
        self._function('double_exponential_with_right_censoring_rng', [y, mu, sigma, event])          

    def eigendecompose(self, A):
        self._function('eigendecompose', [A])          

    def eigendecompose_sym(self, A):
        self._function('eigendecompose_sym', [A])          

    def eigenvalues(self, A):
        self._function('eigenvalues', [A])          

    def eigenvalues_sym(self, A):
        self._function('eigenvalues_sym', [A])          

    def eigenvectors(self, A):
        self._function('eigenvectors', [A])          

    def eigenvectors_sym(self, A):
        self._function('eigenvectors_sym', [A])          

    def erf(self, x):
        self._function('erf', [x])          

    def erfc(self, x):
        self._function('erfc', [x])          

    def exp(self, z):
        self._function('exp', [z])          

    def exp2(self, x):
        self._function('exp2', [x])          

    def exp_mod_normal(self, mu, sigma, lambda_):
        self._function('exp_mod_normal', [mu, sigma, lambda_])          

    def exp_mod_normal_cdf(self, y, mu, sigma, lambda_):
        self._function('exp_mod_normal_cdf', [y, mu, sigma, lambda_])          

    def exp_mod_normal_lccdf(self, y, mu, sigma, lambda_):
        self._function('exp_mod_normal_lccdf', [y, mu, sigma, lambda_])          

    def exp_mod_normal_lcdf(self, y, mu, sigma, lambda_):
        self._function('exp_mod_normal_lcdf', [y, mu, sigma, lambda_])          

    def exp_mod_normal_lpdf(self, y, mu, sigma, lambda_):
        self._function('exp_mod_normal_lpdf', [y, mu, sigma, lambda_])          

    def exp_mod_normal_lupdf(self, y, mu, sigma, lambda_):
        self._function('exp_mod_normal_lupdf', [y, mu, sigma, lambda_])          

    def exp_mod_normal_rng(self, mu, sigma, lambda_):
        self._function('exp_mod_normal_rng', [mu, sigma, lambda_])          

    def exp_mod_normal_with_left_and_right_censoring(self, mu, sigma, lambda_, event_left, event_right):
        self._function('exp_mod_normal_with_left_and_right_censoring', [mu, sigma, lambda_, event_left, event_right])          

    def exp_mod_normal_with_left_and_right_censoring_lpdf(self, y, mu, sigma, lambda_, event_left, event_right):
        self._function('exp_mod_normal_with_left_and_right_censoring_lpdf', [y, mu, sigma, lambda_, event_left, event_right])          

    def exp_mod_normal_with_left_and_right_censoring_rng(self, mu, sigma, lambda_, event_left, event_right):
        self._function('exp_mod_normal_with_left_and_right_censoring_rng', [mu, sigma, lambda_, event_left, event_right])          

    def exp_mod_normal_with_left_censoring(self, mu, sigma, lambda_, event):
        self._function('exp_mod_normal_with_left_censoring', [mu, sigma, lambda_, event])          

    def exp_mod_normal_with_left_censoring_lpdf(self, y, mu, sigma, lambda_, event):
        self._function('exp_mod_normal_with_left_censoring_lpdf', [y, mu, sigma, lambda_, event])          

    def exp_mod_normal_with_left_censoring_rng(self, y, mu, sigma, lambda_, event):
        self._function('exp_mod_normal_with_left_censoring_rng', [y, mu, sigma, lambda_, event])          

    def exp_mod_normal_with_right_censoring(self, mu, sigma, lambda_):
        self._function('exp_mod_normal_with_right_censoring', [mu, sigma, lambda_])          

    def exp_mod_normal_with_right_censoring_lpdf(self, y, mu, sigma, lambda_, event):
        self._function('exp_mod_normal_with_right_censoring_lpdf', [y, mu, sigma, lambda_, event])          

    def exp_mod_normal_with_right_censoring_rng(self, y, mu, sigma, lambda_, event):
        self._function('exp_mod_normal_with_right_censoring_rng', [y, mu, sigma, lambda_, event])          

    def expm1(self, x):
        self._function('expm1', [x])          

    def exponential(self, beta):
        self._function('exponential', [beta])          

    def exponential_cdf(self, y, beta):
        self._function('exponential_cdf', [y, beta])          

    def exponential_lccdf(self, y, beta):
        self._function('exponential_lccdf', [y, beta])          

    def exponential_lcdf(self, y, beta):
        self._function('exponential_lcdf', [y, beta])          

    def exponential_lpdf(self, y, beta):
        self._function('exponential_lpdf', [y, beta])          

    def exponential_lupdf(self, y, beta):
        self._function('exponential_lupdf', [y, beta])          

    def exponential_rng(self, beta):
        self._function('exponential_rng', [beta])          

    def exponential_with_left_and_right_censoring(self, beta, event_left, event_right):
        self._function('exponential_with_left_and_right_censoring', [beta, event_left, event_right])          

    def exponential_with_left_and_right_censoring_lpdf(self, y, beta, event_left, event_right):
        self._function('exponential_with_left_and_right_censoring_lpdf', [y, beta, event_left, event_right])          

    def exponential_with_left_and_right_censoring_rng(self, beta, event_left, event_right):
        self._function('exponential_with_left_and_right_censoring_rng', [beta, event_left, event_right])          

    def exponential_with_left_censoring(self, beta, event):
        self._function('exponential_with_left_censoring', [beta, event])          

    def exponential_with_left_censoring_lpdf(self, y, beta, event):
        self._function('exponential_with_left_censoring_lpdf', [y, beta, event])          

    def exponential_with_left_censoring_rng(self, y, beta, event):
        self._function('exponential_with_left_censoring_rng', [y, beta, event])          

    def exponential_with_right_censoring(self, beta):
        self._function('exponential_with_right_censoring', [beta])          

    def exponential_with_right_censoring_lpdf(self, y, beta, event):
        self._function('exponential_with_right_censoring_lpdf', [y, beta, event])          

    def exponential_with_right_censoring_rng(self, y, beta, event):
        self._function('exponential_with_right_censoring_rng', [y, beta, event])          

    def falling_factorial(self, x, n):
        self._function('falling_factorial', [x, n])          

    def fdim(self, x, y):
        self._function('fdim', [x, y])          

    def fft(self, v):
        self._function('fft', [v])          

    def fft2(self, m):
        self._function('fft2', [m])          

    def floor(self, x):
        self._function('floor', [x])          

    def fma(self, x, y, z):
        self._function('fma', [x, y, z])          

    def fmax(self, x, y):
        self._function('fmax', [x, y])          

    def fmin(self, x, y):
        self._function('fmin', [x, y])          

    def fmod(self, x, y):
        self._function('fmod', [x, y])          

    def frechet(self, alpha, sigma):
        self._function('frechet', [alpha, sigma])          

    def frechet_cdf(self, y, alpha, sigma):
        self._function('frechet_cdf', [y, alpha, sigma])          

    def frechet_lccdf(self, y, alpha, sigma):
        self._function('frechet_lccdf', [y, alpha, sigma])          

    def frechet_lcdf(self, y, alpha, sigma):
        self._function('frechet_lcdf', [y, alpha, sigma])          

    def frechet_lpdf(self, y, alpha, sigma):
        self._function('frechet_lpdf', [y, alpha, sigma])          

    def frechet_lupdf(self, y, alpha, sigma):
        self._function('frechet_lupdf', [y, alpha, sigma])          

    def frechet_rng(self, alpha, sigma):
        self._function('frechet_rng', [alpha, sigma])          

    def frechet_with_left_and_right_censoring(self, alpha, sigma, event_left, event_right):
        self._function('frechet_with_left_and_right_censoring', [alpha, sigma, event_left, event_right])          

    def frechet_with_left_and_right_censoring_lpdf(self, y, alpha, sigma, event_left, event_right):
        self._function('frechet_with_left_and_right_censoring_lpdf', [y, alpha, sigma, event_left, event_right])          

    def frechet_with_left_and_right_censoring_rng(self, alpha, sigma, event_left, event_right):
        self._function('frechet_with_left_and_right_censoring_rng', [alpha, sigma, event_left, event_right])          

    def frechet_with_left_censoring(self, alpha, sigma, event):
        self._function('frechet_with_left_censoring', [alpha, sigma, event])          

    def frechet_with_left_censoring_lpdf(self, y, alpha, sigma, event):
        self._function('frechet_with_left_censoring_lpdf', [y, alpha, sigma, event])          

    def frechet_with_left_censoring_rng(self, y, alpha, sigma, event):
        self._function('frechet_with_left_censoring_rng', [y, alpha, sigma, event])          

    def frechet_with_right_censoring(self, alpha, sigma):
        self._function('frechet_with_right_censoring', [alpha, sigma])          

    def frechet_with_right_censoring_lpdf(self, y, alpha, sigma, event):
        self._function('frechet_with_right_censoring_lpdf', [y, alpha, sigma, event])          

    def frechet_with_right_censoring_rng(self, y, alpha, sigma, event):
        self._function('frechet_with_right_censoring_rng', [y, alpha, sigma, event])          

    def gamma(self, alpha, beta):
        self._function('gamma', [alpha, beta])          

    def gamma_cdf(self, y, alpha, beta):
        self._function('gamma_cdf', [y, alpha, beta])          

    def gamma_lccdf(self, y, alpha, beta):
        self._function('gamma_lccdf', [y, alpha, beta])          

    def gamma_lcdf(self, y, alpha, beta):
        self._function('gamma_lcdf', [y, alpha, beta])          

    def gamma_lpdf(self, y, alpha, beta):
        self._function('gamma_lpdf', [y, alpha, beta])          

    def gamma_lupdf(self, y, alpha, beta):
        self._function('gamma_lupdf', [y, alpha, beta])          

    def gamma_p(self, a, z):
        self._function('gamma_p', [a, z])          

    def gamma_q(self, a, z):
        self._function('gamma_q', [a, z])          

    def gamma_rng(self, alpha, beta):
        self._function('gamma_rng', [alpha, beta])          

    def gamma_with_left_and_right_censoring(self, alpha, beta, event_left, event_right):
        self._function('gamma_with_left_and_right_censoring', [alpha, beta, event_left, event_right])          

    def gamma_with_left_and_right_censoring_lpdf(self, y, alpha, beta, event_left, event_right):
        self._function('gamma_with_left_and_right_censoring_lpdf', [y, alpha, beta, event_left, event_right])          

    def gamma_with_left_and_right_censoring_rng(self, alpha, beta, event_left, event_right):
        self._function('gamma_with_left_and_right_censoring_rng', [alpha, beta, event_left, event_right])          

    def gamma_with_left_censoring(self, alpha, beta, event):
        self._function('gamma_with_left_censoring', [alpha, beta, event])          

    def gamma_with_left_censoring_lpdf(self, y, alpha, beta, event):
        self._function('gamma_with_left_censoring_lpdf', [y, alpha, beta, event])          

    def gamma_with_left_censoring_rng(self, y, alpha, beta, event):
        self._function('gamma_with_left_censoring_rng', [y, alpha, beta, event])          

    def gamma_with_right_censoring(self, alpha, beta):
        self._function('gamma_with_right_censoring', [alpha, beta])          

    def gamma_with_right_censoring_lpdf(self, y, alpha, beta, event):
        self._function('gamma_with_right_censoring_lpdf', [y, alpha, beta, event])          

    def gamma_with_right_censoring_rng(self, y, alpha, beta, event):
        self._function('gamma_with_right_censoring_rng', [y, alpha, beta, event])          

    def gaussian_dlm_obs(self, F, G, V, W, m0, C0):
        self._function('gaussian_dlm_obs', [F, G, V, W, m0, C0])          

    def gaussian_dlm_obs_lpdf(self, y, F, G, V, W, m0, C0):
        self._function('gaussian_dlm_obs_lpdf', [y, F, G, V, W, m0, C0])          

    def gaussian_dlm_obs_lupdf(self, y, F, G, V, W, m0, C0):
        self._function('gaussian_dlm_obs_lupdf', [y, F, G, V, W, m0, C0])          

    def gaussian_dlm_obs_with_left_and_right_censoring(self, F, G, V, W, m0, C0, event_left, event_right):
        self._function('gaussian_dlm_obs_with_left_and_right_censoring', [F, G, V, W, m0, C0, event_left, event_right])          

    def gaussian_dlm_obs_with_left_and_right_censoring_lpdf(self, y, F, G, V, W, m0, C0, event_left, event_right):
        self._function('gaussian_dlm_obs_with_left_and_right_censoring_lpdf', [y, F, G, V, W, m0, C0, event_left, event_right])          

    def gaussian_dlm_obs_with_left_and_right_censoring_rng(self, F, G, V, W, m0, C0, event_left, event_right):
        self._function('gaussian_dlm_obs_with_left_and_right_censoring_rng', [F, G, V, W, m0, C0, event_left, event_right])          

    def gaussian_dlm_obs_with_left_censoring(self, F, G, V, W, m0, C0, event):
        self._function('gaussian_dlm_obs_with_left_censoring', [F, G, V, W, m0, C0, event])          

    def gaussian_dlm_obs_with_left_censoring_lpdf(self, y, F, G, V, W, m0, C0, event):
        self._function('gaussian_dlm_obs_with_left_censoring_lpdf', [y, F, G, V, W, m0, C0, event])          

    def gaussian_dlm_obs_with_left_censoring_rng(self, y, F, G, V, W, m0, C0, event):
        self._function('gaussian_dlm_obs_with_left_censoring_rng', [y, F, G, V, W, m0, C0, event])          

    def gaussian_dlm_obs_with_right_censoring(self, F, G, V, W, m0, C0):
        self._function('gaussian_dlm_obs_with_right_censoring', [F, G, V, W, m0, C0])          

    def gaussian_dlm_obs_with_right_censoring_lpdf(self, y, F, G, V, W, m0, C0, event):
        self._function('gaussian_dlm_obs_with_right_censoring_lpdf', [y, F, G, V, W, m0, C0, event])          

    def gaussian_dlm_obs_with_right_censoring_rng(self, y, F, G, V, W, m0, C0, event):
        self._function('gaussian_dlm_obs_with_right_censoring_rng', [y, F, G, V, W, m0, C0, event])          

    def generalized_inverse(self, A):
        self._function('generalized_inverse', [A])          

    def get_imag(self, z):
        self._function('get_imag', [z])          

    def get_real(self, z):
        self._function('get_real', [z])          

    def gumbel(self, mu, beta):
        self._function('gumbel', [mu, beta])          

    def gumbel_cdf(self, y, mu, beta):
        self._function('gumbel_cdf', [y, mu, beta])          

    def gumbel_lccdf(self, y, mu, beta):
        self._function('gumbel_lccdf', [y, mu, beta])          

    def gumbel_lcdf(self, y, mu, beta):
        self._function('gumbel_lcdf', [y, mu, beta])          

    def gumbel_lpdf(self, y, mu, beta):
        self._function('gumbel_lpdf', [y, mu, beta])          

    def gumbel_lupdf(self, y, mu, beta):
        self._function('gumbel_lupdf', [y, mu, beta])          

    def gumbel_rng(self, mu, beta):
        self._function('gumbel_rng', [mu, beta])          

    def gumbel_with_left_and_right_censoring(self, mu, beta, event_left, event_right):
        self._function('gumbel_with_left_and_right_censoring', [mu, beta, event_left, event_right])          

    def gumbel_with_left_and_right_censoring_lpdf(self, y, mu, beta, event_left, event_right):
        self._function('gumbel_with_left_and_right_censoring_lpdf', [y, mu, beta, event_left, event_right])          

    def gumbel_with_left_and_right_censoring_rng(self, mu, beta, event_left, event_right):
        self._function('gumbel_with_left_and_right_censoring_rng', [mu, beta, event_left, event_right])          

    def gumbel_with_left_censoring(self, mu, beta, event):
        self._function('gumbel_with_left_censoring', [mu, beta, event])          

    def gumbel_with_left_censoring_lpdf(self, y, mu, beta, event):
        self._function('gumbel_with_left_censoring_lpdf', [y, mu, beta, event])          

    def gumbel_with_left_censoring_rng(self, y, mu, beta, event):
        self._function('gumbel_with_left_censoring_rng', [y, mu, beta, event])          

    def gumbel_with_right_censoring(self, mu, beta):
        self._function('gumbel_with_right_censoring', [mu, beta])          

    def gumbel_with_right_censoring_lpdf(self, y, mu, beta, event):
        self._function('gumbel_with_right_censoring_lpdf', [y, mu, beta, event])          

    def gumbel_with_right_censoring_rng(self, y, mu, beta, event):
        self._function('gumbel_with_right_censoring_rng', [y, mu, beta, event])          

    def head(self, v, n):
        self._function('head', [v, n])          

    def hmm_hidden_state_prob(self, log_omega, Gamma, rho):
        self._function('hmm_hidden_state_prob', [log_omega, Gamma, rho])          

    def hmm_latent_rng(self, log_omega, Gamma, rho):
        self._function('hmm_latent_rng', [log_omega, Gamma, rho])          

    def hmm_marginal(self, log_omega, Gamma, rho):
        self._function('hmm_marginal', [log_omega, Gamma, rho])          

    def hypergeometric(self, N, a, b):
        self._function('hypergeometric', [N, a, b])          

    def hypergeometric_lpmf(self, n, N, a, b):
        self._function('hypergeometric_lpmf', [n, N, a, b])          

    def hypergeometric_lupmf(self, n, N, a, b):
        self._function('hypergeometric_lupmf', [n, N, a, b])          

    def hypergeometric_rng(self, N, a, b):
        self._function('hypergeometric_rng', [N, a, b])          

    def hypergeometric_with_left_and_right_censoring(self, N, a, b, event_left, event_right):
        self._function('hypergeometric_with_left_and_right_censoring', [N, a, b, event_left, event_right])          

    def hypergeometric_with_left_and_right_censoring_lpmf(self, n, N, a, b, event_left, event_right):
        self._function('hypergeometric_with_left_and_right_censoring_lpmf', [n, N, a, b, event_left, event_right])          

    def hypergeometric_with_left_and_right_censoring_rng(self, N, a, b, event_left, event_right):
        self._function('hypergeometric_with_left_and_right_censoring_rng', [N, a, b, event_left, event_right])          

    def hypergeometric_with_left_censoring(self, N, a, b, event):
        self._function('hypergeometric_with_left_censoring', [N, a, b, event])          

    def hypergeometric_with_left_censoring_lpmf(self, n, N, a, b, event):
        self._function('hypergeometric_with_left_censoring_lpmf', [n, N, a, b, event])          

    def hypergeometric_with_left_censoring_rng(self, n, N, a, b, event):
        self._function('hypergeometric_with_left_censoring_rng', [n, N, a, b, event])          

    def hypergeometric_with_right_censoring(self, N, a, b):
        self._function('hypergeometric_with_right_censoring', [N, a, b])          

    def hypergeometric_with_right_censoring_lpmf(self, n, N, a, b, event):
        self._function('hypergeometric_with_right_censoring_lpmf', [n, N, a, b, event])          

    def hypergeometric_with_right_censoring_rng(self, n, N, a, b, event):
        self._function('hypergeometric_with_right_censoring_rng', [n, N, a, b, event])          

    def hypot(self, x, y):
        self._function('hypot', [x, y])          

    def identity_matrix(self, k):
        self._function('identity_matrix', [k])          

    def inc_beta(self, alpha, beta, x):
        self._function('inc_beta', [alpha, beta, x])          

    def int_step(self, x):
        self._function('int_step', [x])          

    def integrate_1d(self, integrand, a, b, theta, x_r, x_i):
        self._function('integrate_1d', [integrand, a, b, theta, x_r, x_i])          

    def integrate_ode(self, ode, initial_state, initial_time, times, theta, x_r, x_i):
        self._function('integrate_ode', [ode, initial_state, initial_time, times, theta, x_r, x_i])          

    def integrate_ode_adams(self, ode, initial_state, initial_time, times, theta, x_r, x_i):
        self._function('integrate_ode_adams', [ode, initial_state, initial_time, times, theta, x_r, x_i])          

    def integrate_ode_bdf(self, ode, initial_state, initial_time, times, theta, x_r, x_i):
        self._function('integrate_ode_bdf', [ode, initial_state, initial_time, times, theta, x_r, x_i])          

    def integrate_ode_rk45(self, ode, initial_state, initial_time, times, theta, x_r, x_i):
        self._function('integrate_ode_rk45', [ode, initial_state, initial_time, times, theta, x_r, x_i])          

    def inv(self, x):
        self._function('inv', [x])          

    def inv_chi_square(self, nu):
        self._function('inv_chi_square', [nu])          

    def inv_chi_square_cdf(self, y, nu):
        self._function('inv_chi_square_cdf', [y, nu])          

    def inv_chi_square_lccdf(self, y, nu):
        self._function('inv_chi_square_lccdf', [y, nu])          

    def inv_chi_square_lcdf(self, y, nu):
        self._function('inv_chi_square_lcdf', [y, nu])          

    def inv_chi_square_lpdf(self, y, nu):
        self._function('inv_chi_square_lpdf', [y, nu])          

    def inv_chi_square_lupdf(self, y, nu):
        self._function('inv_chi_square_lupdf', [y, nu])          

    def inv_chi_square_rng(self, nu):
        self._function('inv_chi_square_rng', [nu])          

    def inv_chi_square_with_left_and_right_censoring(self, nu, event_left, event_right):
        self._function('inv_chi_square_with_left_and_right_censoring', [nu, event_left, event_right])          

    def inv_chi_square_with_left_and_right_censoring_lpdf(self, y, nu, event_left, event_right):
        self._function('inv_chi_square_with_left_and_right_censoring_lpdf', [y, nu, event_left, event_right])          

    def inv_chi_square_with_left_and_right_censoring_rng(self, nu, event_left, event_right):
        self._function('inv_chi_square_with_left_and_right_censoring_rng', [nu, event_left, event_right])          

    def inv_chi_square_with_left_censoring(self, nu, event):
        self._function('inv_chi_square_with_left_censoring', [nu, event])          

    def inv_chi_square_with_left_censoring_lpdf(self, y, nu, event):
        self._function('inv_chi_square_with_left_censoring_lpdf', [y, nu, event])          

    def inv_chi_square_with_left_censoring_rng(self, y, nu, event):
        self._function('inv_chi_square_with_left_censoring_rng', [y, nu, event])          

    def inv_chi_square_with_right_censoring(self, nu):
        self._function('inv_chi_square_with_right_censoring', [nu])          

    def inv_chi_square_with_right_censoring_lpdf(self, y, nu, event):
        self._function('inv_chi_square_with_right_censoring_lpdf', [y, nu, event])          

    def inv_chi_square_with_right_censoring_rng(self, y, nu, event):
        self._function('inv_chi_square_with_right_censoring_rng', [y, nu, event])          

    def inv_cloglog(self, x):
        self._function('inv_cloglog', [x])          

    def inv_erfc(self, x):
        self._function('inv_erfc', [x])          

    def inv_fft(self, u):
        self._function('inv_fft', [u])          

    def inv_fft2(self, m):
        self._function('inv_fft2', [m])          

    def inv_gamma(self, alpha, beta):
        self._function('inv_gamma', [alpha, beta])          

    def inv_gamma_cdf(self, y, alpha, beta):
        self._function('inv_gamma_cdf', [y, alpha, beta])          

    def inv_gamma_lccdf(self, y, alpha, beta):
        self._function('inv_gamma_lccdf', [y, alpha, beta])          

    def inv_gamma_lcdf(self, y, alpha, beta):
        self._function('inv_gamma_lcdf', [y, alpha, beta])          

    def inv_gamma_lpdf(self, y, alpha, beta):
        self._function('inv_gamma_lpdf', [y, alpha, beta])          

    def inv_gamma_lupdf(self, y, alpha, beta):
        self._function('inv_gamma_lupdf', [y, alpha, beta])          

    def inv_gamma_rng(self, alpha, beta):
        self._function('inv_gamma_rng', [alpha, beta])          

    def inv_gamma_with_left_and_right_censoring(self, alpha, beta, event_left, event_right):
        self._function('inv_gamma_with_left_and_right_censoring', [alpha, beta, event_left, event_right])          

    def inv_gamma_with_left_and_right_censoring_lpdf(self, y, alpha, beta, event_left, event_right):
        self._function('inv_gamma_with_left_and_right_censoring_lpdf', [y, alpha, beta, event_left, event_right])          

    def inv_gamma_with_left_and_right_censoring_rng(self, alpha, beta, event_left, event_right):
        self._function('inv_gamma_with_left_and_right_censoring_rng', [alpha, beta, event_left, event_right])          

    def inv_gamma_with_left_censoring(self, alpha, beta, event):
        self._function('inv_gamma_with_left_censoring', [alpha, beta, event])          

    def inv_gamma_with_left_censoring_lpdf(self, y, alpha, beta, event):
        self._function('inv_gamma_with_left_censoring_lpdf', [y, alpha, beta, event])          

    def inv_gamma_with_left_censoring_rng(self, y, alpha, beta, event):
        self._function('inv_gamma_with_left_censoring_rng', [y, alpha, beta, event])          

    def inv_gamma_with_right_censoring(self, alpha, beta):
        self._function('inv_gamma_with_right_censoring', [alpha, beta])          

    def inv_gamma_with_right_censoring_lpdf(self, y, alpha, beta, event):
        self._function('inv_gamma_with_right_censoring_lpdf', [y, alpha, beta, event])          

    def inv_gamma_with_right_censoring_rng(self, y, alpha, beta, event):
        self._function('inv_gamma_with_right_censoring_rng', [y, alpha, beta, event])          

    def inv_inc_beta(self, alpha, beta, p):
        self._function('inv_inc_beta', [alpha, beta, p])          

    def inv_logit(self, x):
        self._function('inv_logit', [x])          

    def inv_Phi(self, x):
        self._function('inv_Phi', [x])          

    def inv_sqrt(self, x):
        self._function('inv_sqrt', [x])          

    def inv_square(self, x):
        self._function('inv_square', [x])          

    def inv_wishart(self, nu, Sigma):
        self._function('inv_wishart', [nu, Sigma])          

    def inv_wishart_cholesky(self, nu, L_S):
        self._function('inv_wishart_cholesky', [nu, L_S])          

    def inv_wishart_cholesky_lpdf(self, L_W, nu, L_S):
        self._function('inv_wishart_cholesky_lpdf', [L_W, nu, L_S])          

    def inv_wishart_cholesky_lupdf(self, L_W, nu, L_S):
        self._function('inv_wishart_cholesky_lupdf', [L_W, nu, L_S])          

    def inv_wishart_cholesky_rng(self, nu, L_S):
        self._function('inv_wishart_cholesky_rng', [nu, L_S])          

    def inv_wishart_cholesky_with_left_and_right_censoring(self, nu, L_S, event_left, event_right):
        self._function('inv_wishart_cholesky_with_left_and_right_censoring', [nu, L_S, event_left, event_right])          

    def inv_wishart_cholesky_with_left_and_right_censoring_lpdf(self, L_W, nu, L_S, event_left, event_right):
        self._function('inv_wishart_cholesky_with_left_and_right_censoring_lpdf', [L_W, nu, L_S, event_left, event_right])          

    def inv_wishart_cholesky_with_left_and_right_censoring_rng(self, nu, L_S, event_left, event_right):
        self._function('inv_wishart_cholesky_with_left_and_right_censoring_rng', [nu, L_S, event_left, event_right])          

    def inv_wishart_cholesky_with_left_censoring(self, nu, L_S, event):
        self._function('inv_wishart_cholesky_with_left_censoring', [nu, L_S, event])          

    def inv_wishart_cholesky_with_left_censoring_lpdf(self, L_W, nu, L_S, event):
        self._function('inv_wishart_cholesky_with_left_censoring_lpdf', [L_W, nu, L_S, event])          

    def inv_wishart_cholesky_with_left_censoring_rng(self, L_W, nu, L_S, event):
        self._function('inv_wishart_cholesky_with_left_censoring_rng', [L_W, nu, L_S, event])          

    def inv_wishart_cholesky_with_right_censoring(self, nu, L_S):
        self._function('inv_wishart_cholesky_with_right_censoring', [nu, L_S])          

    def inv_wishart_cholesky_with_right_censoring_lpdf(self, L_W, nu, L_S, event):
        self._function('inv_wishart_cholesky_with_right_censoring_lpdf', [L_W, nu, L_S, event])          

    def inv_wishart_cholesky_with_right_censoring_rng(self, L_W, nu, L_S, event):
        self._function('inv_wishart_cholesky_with_right_censoring_rng', [L_W, nu, L_S, event])          

    def inv_wishart_lpdf(self, W, nu, Sigma):
        self._function('inv_wishart_lpdf', [W, nu, Sigma])          

    def inv_wishart_lupdf(self, W, nu, Sigma):
        self._function('inv_wishart_lupdf', [W, nu, Sigma])          

    def inv_wishart_rng(self, nu, Sigma):
        self._function('inv_wishart_rng', [nu, Sigma])          

    def inv_wishart_with_left_and_right_censoring(self, nu, Sigma, event_left, event_right):
        self._function('inv_wishart_with_left_and_right_censoring', [nu, Sigma, event_left, event_right])          

    def inv_wishart_with_left_and_right_censoring_lpdf(self, W, nu, Sigma, event_left, event_right):
        self._function('inv_wishart_with_left_and_right_censoring_lpdf', [W, nu, Sigma, event_left, event_right])          

    def inv_wishart_with_left_and_right_censoring_rng(self, nu, Sigma, event_left, event_right):
        self._function('inv_wishart_with_left_and_right_censoring_rng', [nu, Sigma, event_left, event_right])          

    def inv_wishart_with_left_censoring(self, nu, Sigma, event):
        self._function('inv_wishart_with_left_censoring', [nu, Sigma, event])          

    def inv_wishart_with_left_censoring_lpdf(self, W, nu, Sigma, event):
        self._function('inv_wishart_with_left_censoring_lpdf', [W, nu, Sigma, event])          

    def inv_wishart_with_left_censoring_rng(self, W, nu, Sigma, event):
        self._function('inv_wishart_with_left_censoring_rng', [W, nu, Sigma, event])          

    def inv_wishart_with_right_censoring(self, nu, Sigma):
        self._function('inv_wishart_with_right_censoring', [nu, Sigma])          

    def inv_wishart_with_right_censoring_lpdf(self, W, nu, Sigma, event):
        self._function('inv_wishart_with_right_censoring_lpdf', [W, nu, Sigma, event])          

    def inv_wishart_with_right_censoring_rng(self, W, nu, Sigma, event):
        self._function('inv_wishart_with_right_censoring_rng', [W, nu, Sigma, event])          

    def inverse(self, A):
        self._function('inverse', [A])          

    def inverse_spd(self, A):
        self._function('inverse_spd', [A])          

    def is_inf(self, x):
        self._function('is_inf', [x])          

    def is_nan(self, x):
        self._function('is_nan', [x])          

    def lambert_w0(self, x):
        self._function('lambert_w0', [x])          

    def lambert_wm1(self, x):
        self._function('lambert_wm1', [x])          

    def lbeta(self, alpha, beta):
        self._function('lbeta', [alpha, beta])          

    def lchoose(self, x, y):
        self._function('lchoose', [x, y])          

    def ldexp(self, x, y):
        self._function('ldexp', [x, y])          

    def lgamma(self, x):
        self._function('lgamma', [x])          

    def linspaced_array(self, n, lower, upper):
        self._function('linspaced_array', [n, lower, upper])          

    def linspaced_int_array(self, n, lower, upper):
        self._function('linspaced_int_array', [n, lower, upper])          

    def linspaced_row_vector(self, n, lower, upper):
        self._function('linspaced_row_vector', [n, lower, upper])          

    def linspaced_vector(self, n, lower, upper):
        self._function('linspaced_vector', [n, lower, upper])          

    def lkj_corr(self, eta):
        self._function('lkj_corr', [eta])          

    def lkj_corr_cholesky(self, eta):
        self._function('lkj_corr_cholesky', [eta])          

    def lkj_corr_cholesky_lpdf(self, L, eta):
        self._function('lkj_corr_cholesky_lpdf', [L, eta])          

    def lkj_corr_cholesky_lupdf(self, L, eta):
        self._function('lkj_corr_cholesky_lupdf', [L, eta])          

    def lkj_corr_cholesky_rng(self, K, eta):
        self._function('lkj_corr_cholesky_rng', [K, eta])          

    def lkj_corr_cholesky_with_left_and_right_censoring(self, eta, event_left, event_right):
        self._function('lkj_corr_cholesky_with_left_and_right_censoring', [eta, event_left, event_right])          

    def lkj_corr_cholesky_with_left_and_right_censoring_lpdf(self, L, eta, event_left, event_right):
        self._function('lkj_corr_cholesky_with_left_and_right_censoring_lpdf', [L, eta, event_left, event_right])          

    def lkj_corr_cholesky_with_left_and_right_censoring_rng(self, eta, event_left, event_right):
        self._function('lkj_corr_cholesky_with_left_and_right_censoring_rng', [eta, event_left, event_right])          

    def lkj_corr_cholesky_with_left_censoring(self, eta, event):
        self._function('lkj_corr_cholesky_with_left_censoring', [eta, event])          

    def lkj_corr_cholesky_with_left_censoring_lpdf(self, L, eta, event):
        self._function('lkj_corr_cholesky_with_left_censoring_lpdf', [L, eta, event])          

    def lkj_corr_cholesky_with_left_censoring_rng(self, L, eta, event):
        self._function('lkj_corr_cholesky_with_left_censoring_rng', [L, eta, event])          

    def lkj_corr_cholesky_with_right_censoring(self, eta):
        self._function('lkj_corr_cholesky_with_right_censoring', [eta])          

    def lkj_corr_cholesky_with_right_censoring_lpdf(self, L, eta, event):
        self._function('lkj_corr_cholesky_with_right_censoring_lpdf', [L, eta, event])          

    def lkj_corr_cholesky_with_right_censoring_rng(self, L, eta, event):
        self._function('lkj_corr_cholesky_with_right_censoring_rng', [L, eta, event])          

    def lkj_corr_lpdf(self, y, eta):
        self._function('lkj_corr_lpdf', [y, eta])          

    def lkj_corr_lupdf(self, y, eta):
        self._function('lkj_corr_lupdf', [y, eta])          

    def lkj_corr_rng(self, K, eta):
        self._function('lkj_corr_rng', [K, eta])          

    def lkj_corr_with_left_and_right_censoring(self, eta, event_left, event_right):
        self._function('lkj_corr_with_left_and_right_censoring', [eta, event_left, event_right])          

    def lkj_corr_with_left_and_right_censoring_lpdf(self, y, eta, event_left, event_right):
        self._function('lkj_corr_with_left_and_right_censoring_lpdf', [y, eta, event_left, event_right])          

    def lkj_corr_with_left_and_right_censoring_rng(self, eta, event_left, event_right):
        self._function('lkj_corr_with_left_and_right_censoring_rng', [eta, event_left, event_right])          

    def lkj_corr_with_left_censoring(self, eta, event):
        self._function('lkj_corr_with_left_censoring', [eta, event])          

    def lkj_corr_with_left_censoring_lpdf(self, y, eta, event):
        self._function('lkj_corr_with_left_censoring_lpdf', [y, eta, event])          

    def lkj_corr_with_left_censoring_rng(self, y, eta, event):
        self._function('lkj_corr_with_left_censoring_rng', [y, eta, event])          

    def lkj_corr_with_right_censoring(self, eta):
        self._function('lkj_corr_with_right_censoring', [eta])          

    def lkj_corr_with_right_censoring_lpdf(self, y, eta, event):
        self._function('lkj_corr_with_right_censoring_lpdf', [y, eta, event])          

    def lkj_corr_with_right_censoring_rng(self, y, eta, event):
        self._function('lkj_corr_with_right_censoring_rng', [y, eta, event])          

    def lmgamma(self, n, x):
        self._function('lmgamma', [n, x])          

    def lmultiply(self, x, y):
        self._function('lmultiply', [x, y])          

    def log(self, z):
        self._function('log', [z])          

    def log10(self, z):
        self._function('log10', [z])          

    def log1m(self, x):
        self._function('log1m', [x])          

    def log1m_exp(self, x):
        self._function('log1m_exp', [x])          

    def log1m_inv_logit(self, x):
        self._function('log1m_inv_logit', [x])          

    def log1p(self, x):
        self._function('log1p', [x])          

    def log1p_exp(self, x):
        self._function('log1p_exp', [x])          

    def log2(self, x):
        self._function('log2', [x])          

    def log_determinant(self, A):
        self._function('log_determinant', [A])          

    def log_diff_exp(self, x, y):
        self._function('log_diff_exp', [x, y])          

    def log_falling_factorial(self, x, n):
        self._function('log_falling_factorial', [x, n])          

    def log_inv_logit(self, x):
        self._function('log_inv_logit', [x])          

    def log_inv_logit_diff(self, x, y):
        self._function('log_inv_logit_diff', [x, y])          

    def log_mix(self, theta, lp1, lp2):
        self._function('log_mix', [theta, lp1, lp2])          

    def log_modified_bessel_first_kind(self, v, z):
        self._function('log_modified_bessel_first_kind', [v, z])          

    def log_rising_factorial(self, x, n):
        self._function('log_rising_factorial', [x, n])          

    def log_softmax(self, x):
        self._function('log_softmax', [x])          

    def log_sum_exp(self, x):
        self._function('log_sum_exp', [x])          

    def logistic(self, mu, sigma):
        self._function('logistic', [mu, sigma])          

    def logistic_cdf(self, y, mu, sigma):
        self._function('logistic_cdf', [y, mu, sigma])          

    def logistic_lccdf(self, y, mu, sigma):
        self._function('logistic_lccdf', [y, mu, sigma])          

    def logistic_lcdf(self, y, mu, sigma):
        self._function('logistic_lcdf', [y, mu, sigma])          

    def logistic_lpdf(self, y, mu, sigma):
        self._function('logistic_lpdf', [y, mu, sigma])          

    def logistic_lupdf(self, y, mu, sigma):
        self._function('logistic_lupdf', [y, mu, sigma])          

    def logistic_rng(self, mu, sigma):
        self._function('logistic_rng', [mu, sigma])          

    def logistic_with_left_and_right_censoring(self, mu, sigma, event_left, event_right):
        self._function('logistic_with_left_and_right_censoring', [mu, sigma, event_left, event_right])          

    def logistic_with_left_and_right_censoring_lpdf(self, y, mu, sigma, event_left, event_right):
        self._function('logistic_with_left_and_right_censoring_lpdf', [y, mu, sigma, event_left, event_right])          

    def logistic_with_left_and_right_censoring_rng(self, mu, sigma, event_left, event_right):
        self._function('logistic_with_left_and_right_censoring_rng', [mu, sigma, event_left, event_right])          

    def logistic_with_left_censoring(self, mu, sigma, event):
        self._function('logistic_with_left_censoring', [mu, sigma, event])          

    def logistic_with_left_censoring_lpdf(self, y, mu, sigma, event):
        self._function('logistic_with_left_censoring_lpdf', [y, mu, sigma, event])          

    def logistic_with_left_censoring_rng(self, y, mu, sigma, event):
        self._function('logistic_with_left_censoring_rng', [y, mu, sigma, event])          

    def logistic_with_right_censoring(self, mu, sigma):
        self._function('logistic_with_right_censoring', [mu, sigma])          

    def logistic_with_right_censoring_lpdf(self, y, mu, sigma, event):
        self._function('logistic_with_right_censoring_lpdf', [y, mu, sigma, event])          

    def logistic_with_right_censoring_rng(self, y, mu, sigma, event):
        self._function('logistic_with_right_censoring_rng', [y, mu, sigma, event])          

    def logit(self, x):
        self._function('logit', [x])          

    def loglogistic(self, alpha, beta):
        self._function('loglogistic', [alpha, beta])          

    def loglogistic_cdf(self, y, alpha, beta):
        self._function('loglogistic_cdf', [y, alpha, beta])          

    def loglogistic_lpdf(self, y, alpha, beta):
        self._function('loglogistic_lpdf', [y, alpha, beta])          

    def loglogistic_rng(self, mu, sigma):
        self._function('loglogistic_rng', [mu, sigma])          

    def loglogistic_with_left_and_right_censoring(self, alpha, beta, event_left, event_right):
        self._function('loglogistic_with_left_and_right_censoring', [alpha, beta, event_left, event_right])          

    def loglogistic_with_left_and_right_censoring_lpdf(self, y, alpha, beta, event_left, event_right):
        self._function('loglogistic_with_left_and_right_censoring_lpdf', [y, alpha, beta, event_left, event_right])          

    def loglogistic_with_left_and_right_censoring_rng(self, alpha, beta, event_left, event_right):
        self._function('loglogistic_with_left_and_right_censoring_rng', [alpha, beta, event_left, event_right])          

    def loglogistic_with_left_censoring(self, alpha, beta, event):
        self._function('loglogistic_with_left_censoring', [alpha, beta, event])          

    def loglogistic_with_left_censoring_lpdf(self, y, alpha, beta, event):
        self._function('loglogistic_with_left_censoring_lpdf', [y, alpha, beta, event])          

    def loglogistic_with_left_censoring_rng(self, y, alpha, beta, event):
        self._function('loglogistic_with_left_censoring_rng', [y, alpha, beta, event])          

    def loglogistic_with_right_censoring(self, alpha, beta):
        self._function('loglogistic_with_right_censoring', [alpha, beta])          

    def loglogistic_with_right_censoring_lpdf(self, y, alpha, beta, event):
        self._function('loglogistic_with_right_censoring_lpdf', [y, alpha, beta, event])          

    def loglogistic_with_right_censoring_rng(self, y, alpha, beta, event):
        self._function('loglogistic_with_right_censoring_rng', [y, alpha, beta, event])          

    def lognormal(self, mu, sigma):
        self._function('lognormal', [mu, sigma])          

    def lognormal_cdf(self, y, mu, sigma):
        self._function('lognormal_cdf', [y, mu, sigma])          

    def lognormal_lccdf(self, y, mu, sigma):
        self._function('lognormal_lccdf', [y, mu, sigma])          

    def lognormal_lcdf(self, y, mu, sigma):
        self._function('lognormal_lcdf', [y, mu, sigma])          

    def lognormal_lpdf(self, y, mu, sigma):
        self._function('lognormal_lpdf', [y, mu, sigma])          

    def lognormal_lupdf(self, y, mu, sigma):
        self._function('lognormal_lupdf', [y, mu, sigma])          

    def lognormal_rng(self, mu, sigma):
        self._function('lognormal_rng', [mu, sigma])          

    def lognormal_with_left_and_right_censoring(self, mu, sigma, event_left, event_right):
        self._function('lognormal_with_left_and_right_censoring', [mu, sigma, event_left, event_right])          

    def lognormal_with_left_and_right_censoring_lpdf(self, y, mu, sigma, event_left, event_right):
        self._function('lognormal_with_left_and_right_censoring_lpdf', [y, mu, sigma, event_left, event_right])          

    def lognormal_with_left_and_right_censoring_rng(self, mu, sigma, event_left, event_right):
        self._function('lognormal_with_left_and_right_censoring_rng', [mu, sigma, event_left, event_right])          

    def lognormal_with_left_censoring(self, mu, sigma, event):
        self._function('lognormal_with_left_censoring', [mu, sigma, event])          

    def lognormal_with_left_censoring_lpdf(self, y, mu, sigma, event):
        self._function('lognormal_with_left_censoring_lpdf', [y, mu, sigma, event])          

    def lognormal_with_left_censoring_rng(self, y, mu, sigma, event):
        self._function('lognormal_with_left_censoring_rng', [y, mu, sigma, event])          

    def lognormal_with_right_censoring(self, mu, sigma):
        self._function('lognormal_with_right_censoring', [mu, sigma])          

    def lognormal_with_right_censoring_lpdf(self, y, mu, sigma, event):
        self._function('lognormal_with_right_censoring_lpdf', [y, mu, sigma, event])          

    def lognormal_with_right_censoring_rng(self, y, mu, sigma, event):
        self._function('lognormal_with_right_censoring_rng', [y, mu, sigma, event])          

    def matrix_exp(self, A):
        self._function('matrix_exp', [A])          

    def matrix_exp_multiply(self, A, B):
        self._function('matrix_exp_multiply', [A, B])          

    def matrix_power(self, A, B):
        self._function('matrix_power', [A, B])          

    def max(self, x):
        self._function('max', [x])          

    def mdivide_left_spd(self, A, b):
        self._function('mdivide_left_spd', [A, b])          

    def mdivide_left_tri_low(self, A, b):
        self._function('mdivide_left_tri_low', [A, b])          

    def mdivide_right_spd(self, b, A):
        self._function('mdivide_right_spd', [b, A])          

    def mdivide_right_tri_low(self, b, A):
        self._function('mdivide_right_tri_low', [b, A])          

    def mean(self, x):
        self._function('mean', [x])          

    def min(self, x):
        self._function('min', [x])          

    def modified_bessel_first_kind(self, v, z):
        self._function('modified_bessel_first_kind', [v, z])          

    def modified_bessel_second_kind(self, v, z):
        self._function('modified_bessel_second_kind', [v, z])          

    def multi_gp(self, Sigma, w):
        self._function('multi_gp', [Sigma, w])          

    def multi_gp_cholesky(self, L, w):
        self._function('multi_gp_cholesky', [L, w])          

    def multi_gp_cholesky_lpdf(self, y, L, w):
        self._function('multi_gp_cholesky_lpdf', [y, L, w])          

    def multi_gp_cholesky_lupdf(self, y, L, w):
        self._function('multi_gp_cholesky_lupdf', [y, L, w])          

    def multi_gp_cholesky_with_left_and_right_censoring(self, L, w, event_left, event_right):
        self._function('multi_gp_cholesky_with_left_and_right_censoring', [L, w, event_left, event_right])          

    def multi_gp_cholesky_with_left_and_right_censoring_lpdf(self, y, L, w, event_left, event_right):
        self._function('multi_gp_cholesky_with_left_and_right_censoring_lpdf', [y, L, w, event_left, event_right])          

    def multi_gp_cholesky_with_left_and_right_censoring_rng(self, L, w, event_left, event_right):
        self._function('multi_gp_cholesky_with_left_and_right_censoring_rng', [L, w, event_left, event_right])          

    def multi_gp_cholesky_with_left_censoring(self, L, w, event):
        self._function('multi_gp_cholesky_with_left_censoring', [L, w, event])          

    def multi_gp_cholesky_with_left_censoring_lpdf(self, y, L, w, event):
        self._function('multi_gp_cholesky_with_left_censoring_lpdf', [y, L, w, event])          

    def multi_gp_cholesky_with_left_censoring_rng(self, y, L, w, event):
        self._function('multi_gp_cholesky_with_left_censoring_rng', [y, L, w, event])          

    def multi_gp_cholesky_with_right_censoring(self, L, w):
        self._function('multi_gp_cholesky_with_right_censoring', [L, w])          

    def multi_gp_cholesky_with_right_censoring_lpdf(self, y, L, w, event):
        self._function('multi_gp_cholesky_with_right_censoring_lpdf', [y, L, w, event])          

    def multi_gp_cholesky_with_right_censoring_rng(self, y, L, w, event):
        self._function('multi_gp_cholesky_with_right_censoring_rng', [y, L, w, event])          

    def multi_gp_lpdf(self, y, Sigma, w):
        self._function('multi_gp_lpdf', [y, Sigma, w])          

    def multi_gp_lupdf(self, y, Sigma, w):
        self._function('multi_gp_lupdf', [y, Sigma, w])          

    def multi_gp_with_left_and_right_censoring(self, Sigma, w, event_left, event_right):
        self._function('multi_gp_with_left_and_right_censoring', [Sigma, w, event_left, event_right])          

    def multi_gp_with_left_and_right_censoring_lpdf(self, y, Sigma, w, event_left, event_right):
        self._function('multi_gp_with_left_and_right_censoring_lpdf', [y, Sigma, w, event_left, event_right])          

    def multi_gp_with_left_and_right_censoring_rng(self, Sigma, w, event_left, event_right):
        self._function('multi_gp_with_left_and_right_censoring_rng', [Sigma, w, event_left, event_right])          

    def multi_gp_with_left_censoring(self, Sigma, w, event):
        self._function('multi_gp_with_left_censoring', [Sigma, w, event])          

    def multi_gp_with_left_censoring_lpdf(self, y, Sigma, w, event):
        self._function('multi_gp_with_left_censoring_lpdf', [y, Sigma, w, event])          

    def multi_gp_with_left_censoring_rng(self, y, Sigma, w, event):
        self._function('multi_gp_with_left_censoring_rng', [y, Sigma, w, event])          

    def multi_gp_with_right_censoring(self, Sigma, w):
        self._function('multi_gp_with_right_censoring', [Sigma, w])          

    def multi_gp_with_right_censoring_lpdf(self, y, Sigma, w, event):
        self._function('multi_gp_with_right_censoring_lpdf', [y, Sigma, w, event])          

    def multi_gp_with_right_censoring_rng(self, y, Sigma, w, event):
        self._function('multi_gp_with_right_censoring_rng', [y, Sigma, w, event])          

    def multi_normal(self, mu, Sigma):
        self._function('multi_normal', [mu, Sigma])          

    def multi_normal_cholesky(self, mu, L):
        self._function('multi_normal_cholesky', [mu, L])          

    def multi_normal_cholesky_lpdf(self, y, mu, L):
        self._function('multi_normal_cholesky_lpdf', [y, mu, L])          

    def multi_normal_cholesky_lupdf(self, y, mu, L):
        self._function('multi_normal_cholesky_lupdf', [y, mu, L])          

    def multi_normal_cholesky_rng(self, mu, L):
        self._function('multi_normal_cholesky_rng', [mu, L])          

    def multi_normal_cholesky_with_left_and_right_censoring(self, mu, L, event_left, event_right):
        self._function('multi_normal_cholesky_with_left_and_right_censoring', [mu, L, event_left, event_right])          

    def multi_normal_cholesky_with_left_and_right_censoring_lpdf(self, y, mu, L, event_left, event_right):
        self._function('multi_normal_cholesky_with_left_and_right_censoring_lpdf', [y, mu, L, event_left, event_right])          

    def multi_normal_cholesky_with_left_and_right_censoring_rng(self, mu, L, event_left, event_right):
        self._function('multi_normal_cholesky_with_left_and_right_censoring_rng', [mu, L, event_left, event_right])          

    def multi_normal_cholesky_with_left_censoring(self, mu, L, event):
        self._function('multi_normal_cholesky_with_left_censoring', [mu, L, event])          

    def multi_normal_cholesky_with_left_censoring_lpdf(self, y, mu, L, event):
        self._function('multi_normal_cholesky_with_left_censoring_lpdf', [y, mu, L, event])          

    def multi_normal_cholesky_with_left_censoring_rng(self, y, mu, L, event):
        self._function('multi_normal_cholesky_with_left_censoring_rng', [y, mu, L, event])          

    def multi_normal_cholesky_with_right_censoring(self, mu, L):
        self._function('multi_normal_cholesky_with_right_censoring', [mu, L])          

    def multi_normal_cholesky_with_right_censoring_lpdf(self, y, mu, L, event):
        self._function('multi_normal_cholesky_with_right_censoring_lpdf', [y, mu, L, event])          

    def multi_normal_cholesky_with_right_censoring_rng(self, y, mu, L, event):
        self._function('multi_normal_cholesky_with_right_censoring_rng', [y, mu, L, event])          

    def multi_normal_lpdf(self, y, mu, Sigma):
        self._function('multi_normal_lpdf', [y, mu, Sigma])          

    def multi_normal_lupdf(self, y, mu, Sigma):
        self._function('multi_normal_lupdf', [y, mu, Sigma])          

    def multi_normal_prec(self, mu, Omega):
        self._function('multi_normal_prec', [mu, Omega])          

    def multi_normal_prec_lpdf(self, y, mu, Omega):
        self._function('multi_normal_prec_lpdf', [y, mu, Omega])          

    def multi_normal_prec_lupdf(self, y, mu, Omega):
        self._function('multi_normal_prec_lupdf', [y, mu, Omega])          

    def multi_normal_prec_with_left_and_right_censoring(self, mu, Omega, event_left, event_right):
        self._function('multi_normal_prec_with_left_and_right_censoring', [mu, Omega, event_left, event_right])          

    def multi_normal_prec_with_left_and_right_censoring_lpdf(self, y, mu, Omega, event_left, event_right):
        self._function('multi_normal_prec_with_left_and_right_censoring_lpdf', [y, mu, Omega, event_left, event_right])          

    def multi_normal_prec_with_left_and_right_censoring_rng(self, mu, Omega, event_left, event_right):
        self._function('multi_normal_prec_with_left_and_right_censoring_rng', [mu, Omega, event_left, event_right])          

    def multi_normal_prec_with_left_censoring(self, mu, Omega, event):
        self._function('multi_normal_prec_with_left_censoring', [mu, Omega, event])          

    def multi_normal_prec_with_left_censoring_lpdf(self, y, mu, Omega, event):
        self._function('multi_normal_prec_with_left_censoring_lpdf', [y, mu, Omega, event])          

    def multi_normal_prec_with_left_censoring_rng(self, y, mu, Omega, event):
        self._function('multi_normal_prec_with_left_censoring_rng', [y, mu, Omega, event])          

    def multi_normal_prec_with_right_censoring(self, mu, Omega):
        self._function('multi_normal_prec_with_right_censoring', [mu, Omega])          

    def multi_normal_prec_with_right_censoring_lpdf(self, y, mu, Omega, event):
        self._function('multi_normal_prec_with_right_censoring_lpdf', [y, mu, Omega, event])          

    def multi_normal_prec_with_right_censoring_rng(self, y, mu, Omega, event):
        self._function('multi_normal_prec_with_right_censoring_rng', [y, mu, Omega, event])          

    def multi_normal_rng(self, mu, Sigma):
        self._function('multi_normal_rng', [mu, Sigma])          

    def multi_normal_with_left_and_right_censoring(self, mu, Sigma, event_left, event_right):
        self._function('multi_normal_with_left_and_right_censoring', [mu, Sigma, event_left, event_right])          

    def multi_normal_with_left_and_right_censoring_lpdf(self, y, mu, Sigma, event_left, event_right):
        self._function('multi_normal_with_left_and_right_censoring_lpdf', [y, mu, Sigma, event_left, event_right])          

    def multi_normal_with_left_and_right_censoring_rng(self, mu, Sigma, event_left, event_right):
        self._function('multi_normal_with_left_and_right_censoring_rng', [mu, Sigma, event_left, event_right])          

    def multi_normal_with_left_censoring(self, mu, Sigma, event):
        self._function('multi_normal_with_left_censoring', [mu, Sigma, event])          

    def multi_normal_with_left_censoring_lpdf(self, y, mu, Sigma, event):
        self._function('multi_normal_with_left_censoring_lpdf', [y, mu, Sigma, event])          

    def multi_normal_with_left_censoring_rng(self, y, mu, Sigma, event):
        self._function('multi_normal_with_left_censoring_rng', [y, mu, Sigma, event])          

    def multi_normal_with_right_censoring(self, mu, Sigma):
        self._function('multi_normal_with_right_censoring', [mu, Sigma])          

    def multi_normal_with_right_censoring_lpdf(self, y, mu, Sigma, event):
        self._function('multi_normal_with_right_censoring_lpdf', [y, mu, Sigma, event])          

    def multi_normal_with_right_censoring_rng(self, y, mu, Sigma, event):
        self._function('multi_normal_with_right_censoring_rng', [y, mu, Sigma, event])          

    def multi_student_cholesky_t_rng(self, nu, mu, L):
        self._function('multi_student_cholesky_t_rng', [nu, mu, L])          

    def multi_student_t(self, nu, mu, Sigma):
        self._function('multi_student_t', [nu, mu, Sigma])          

    def multi_student_t_cholesky(self, nu, mu, L):
        self._function('multi_student_t_cholesky', [nu, mu, L])          

    def multi_student_t_cholesky_lpdf(self, y, nu, mu, L):
        self._function('multi_student_t_cholesky_lpdf', [y, nu, mu, L])          

    def multi_student_t_cholesky_lupdf(self, y, nu, mu, L):
        self._function('multi_student_t_cholesky_lupdf', [y, nu, mu, L])          

    def multi_student_t_cholesky_rng(self, nu, mu, L):
        self._function('multi_student_t_cholesky_rng', [nu, mu, L])          

    def multi_student_t_cholesky_with_left_and_right_censoring(self, nu, mu, L, event_left, event_right):
        self._function('multi_student_t_cholesky_with_left_and_right_censoring', [nu, mu, L, event_left, event_right])          

    def multi_student_t_cholesky_with_left_and_right_censoring_lpdf(self, y, nu, mu, L, event_left, event_right):
        self._function('multi_student_t_cholesky_with_left_and_right_censoring_lpdf', [y, nu, mu, L, event_left, event_right])          

    def multi_student_t_cholesky_with_left_and_right_censoring_rng(self, nu, mu, L, event_left, event_right):
        self._function('multi_student_t_cholesky_with_left_and_right_censoring_rng', [nu, mu, L, event_left, event_right])          

    def multi_student_t_cholesky_with_left_censoring(self, nu, mu, L, event):
        self._function('multi_student_t_cholesky_with_left_censoring', [nu, mu, L, event])          

    def multi_student_t_cholesky_with_left_censoring_lpdf(self, y, nu, mu, L, event):
        self._function('multi_student_t_cholesky_with_left_censoring_lpdf', [y, nu, mu, L, event])          

    def multi_student_t_cholesky_with_left_censoring_rng(self, y, nu, mu, L, event):
        self._function('multi_student_t_cholesky_with_left_censoring_rng', [y, nu, mu, L, event])          

    def multi_student_t_cholesky_with_right_censoring(self, nu, mu, L):
        self._function('multi_student_t_cholesky_with_right_censoring', [nu, mu, L])          

    def multi_student_t_cholesky_with_right_censoring_lpdf(self, y, nu, mu, L, event):
        self._function('multi_student_t_cholesky_with_right_censoring_lpdf', [y, nu, mu, L, event])          

    def multi_student_t_cholesky_with_right_censoring_rng(self, y, nu, mu, L, event):
        self._function('multi_student_t_cholesky_with_right_censoring_rng', [y, nu, mu, L, event])          

    def multi_student_t_lpdf(self, y, nu, mu, Sigma):
        self._function('multi_student_t_lpdf', [y, nu, mu, Sigma])          

    def multi_student_t_lupdf(self, y, nu, mu, Sigma):
        self._function('multi_student_t_lupdf', [y, nu, mu, Sigma])          

    def multi_student_t_rng(self, nu, mu, Sigma):
        self._function('multi_student_t_rng', [nu, mu, Sigma])          

    def multi_student_t_with_left_and_right_censoring(self, nu, mu, Sigma, event_left, event_right):
        self._function('multi_student_t_with_left_and_right_censoring', [nu, mu, Sigma, event_left, event_right])          

    def multi_student_t_with_left_and_right_censoring_lpdf(self, y, nu, mu, Sigma, event_left, event_right):
        self._function('multi_student_t_with_left_and_right_censoring_lpdf', [y, nu, mu, Sigma, event_left, event_right])          

    def multi_student_t_with_left_and_right_censoring_rng(self, nu, mu, Sigma, event_left, event_right):
        self._function('multi_student_t_with_left_and_right_censoring_rng', [nu, mu, Sigma, event_left, event_right])          

    def multi_student_t_with_left_censoring(self, nu, mu, Sigma, event):
        self._function('multi_student_t_with_left_censoring', [nu, mu, Sigma, event])          

    def multi_student_t_with_left_censoring_lpdf(self, y, nu, mu, Sigma, event):
        self._function('multi_student_t_with_left_censoring_lpdf', [y, nu, mu, Sigma, event])          

    def multi_student_t_with_left_censoring_rng(self, y, nu, mu, Sigma, event):
        self._function('multi_student_t_with_left_censoring_rng', [y, nu, mu, Sigma, event])          

    def multi_student_t_with_right_censoring(self, nu, mu, Sigma):
        self._function('multi_student_t_with_right_censoring', [nu, mu, Sigma])          

    def multi_student_t_with_right_censoring_lpdf(self, y, nu, mu, Sigma, event):
        self._function('multi_student_t_with_right_censoring_lpdf', [y, nu, mu, Sigma, event])          

    def multi_student_t_with_right_censoring_rng(self, y, nu, mu, Sigma, event):
        self._function('multi_student_t_with_right_censoring_rng', [y, nu, mu, Sigma, event])          

    def multinomial(self, theta):
        self._function('multinomial', [theta])          

    def multinomial_logit(self, gamma):
        self._function('multinomial_logit', [gamma])          

    def multinomial_logit_lpmf(self, y, gamma):
        self._function('multinomial_logit_lpmf', [y, gamma])          

    def multinomial_logit_lupmf(self, y, gamma):
        self._function('multinomial_logit_lupmf', [y, gamma])          

    def multinomial_logit_rng(self, gamma, N):
        self._function('multinomial_logit_rng', [gamma, N])          

    def multinomial_logit_with_left_and_right_censoring(self, gamma, event_left, event_right):
        self._function('multinomial_logit_with_left_and_right_censoring', [gamma, event_left, event_right])          

    def multinomial_logit_with_left_and_right_censoring_lpmf(self, y, gamma, event_left, event_right):
        self._function('multinomial_logit_with_left_and_right_censoring_lpmf', [y, gamma, event_left, event_right])          

    def multinomial_logit_with_left_and_right_censoring_rng(self, gamma, event_left, event_right):
        self._function('multinomial_logit_with_left_and_right_censoring_rng', [gamma, event_left, event_right])          

    def multinomial_logit_with_left_censoring(self, gamma, event):
        self._function('multinomial_logit_with_left_censoring', [gamma, event])          

    def multinomial_logit_with_left_censoring_lpmf(self, y, gamma, event):
        self._function('multinomial_logit_with_left_censoring_lpmf', [y, gamma, event])          

    def multinomial_logit_with_left_censoring_rng(self, y, gamma, event):
        self._function('multinomial_logit_with_left_censoring_rng', [y, gamma, event])          

    def multinomial_logit_with_right_censoring(self, gamma):
        self._function('multinomial_logit_with_right_censoring', [gamma])          

    def multinomial_logit_with_right_censoring_lpmf(self, y, gamma, event):
        self._function('multinomial_logit_with_right_censoring_lpmf', [y, gamma, event])          

    def multinomial_logit_with_right_censoring_rng(self, y, gamma, event):
        self._function('multinomial_logit_with_right_censoring_rng', [y, gamma, event])          

    def multinomial_lpmf(self, y, theta):
        self._function('multinomial_lpmf', [y, theta])          

    def multinomial_lupmf(self, y, theta):
        self._function('multinomial_lupmf', [y, theta])          

    def multinomial_rng(self, theta, N):
        self._function('multinomial_rng', [theta, N])          

    def multinomial_with_left_and_right_censoring(self, theta, event_left, event_right):
        self._function('multinomial_with_left_and_right_censoring', [theta, event_left, event_right])          

    def multinomial_with_left_and_right_censoring_lpmf(self, y, theta, event_left, event_right):
        self._function('multinomial_with_left_and_right_censoring_lpmf', [y, theta, event_left, event_right])          

    def multinomial_with_left_and_right_censoring_rng(self, theta, event_left, event_right):
        self._function('multinomial_with_left_and_right_censoring_rng', [theta, event_left, event_right])          

    def multinomial_with_left_censoring(self, theta, event):
        self._function('multinomial_with_left_censoring', [theta, event])          

    def multinomial_with_left_censoring_lpmf(self, y, theta, event):
        self._function('multinomial_with_left_censoring_lpmf', [y, theta, event])          

    def multinomial_with_left_censoring_rng(self, y, theta, event):
        self._function('multinomial_with_left_censoring_rng', [y, theta, event])          

    def multinomial_with_right_censoring(self, theta):
        self._function('multinomial_with_right_censoring', [theta])          

    def multinomial_with_right_censoring_lpmf(self, y, theta, event):
        self._function('multinomial_with_right_censoring_lpmf', [y, theta, event])          

    def multinomial_with_right_censoring_rng(self, y, theta, event):
        self._function('multinomial_with_right_censoring_rng', [y, theta, event])          

    def multiply_lower_tri_self_transpose(self, x):
        self._function('multiply_lower_tri_self_transpose', [x])          

    def neg_binomial(self, alpha, beta):
        self._function('neg_binomial', [alpha, beta])          

    def neg_binomial_2(self, mu, phi):
        self._function('neg_binomial_2', [mu, phi])          

    def neg_binomial_2_cdf(self, n, mu, phi):
        self._function('neg_binomial_2_cdf', [n, mu, phi])          

    def neg_binomial_2_lccdf(self, n, mu, phi):
        self._function('neg_binomial_2_lccdf', [n, mu, phi])          

    def neg_binomial_2_lcdf(self, n, mu, phi):
        self._function('neg_binomial_2_lcdf', [n, mu, phi])          

    def neg_binomial_2_log(self, eta, phi):
        self._function('neg_binomial_2_log', [eta, phi])          

    def neg_binomial_2_log_glm(self, x, alpha, beta, phi):
        self._function('neg_binomial_2_log_glm', [x, alpha, beta, phi])          

    def neg_binomial_2_log_glm_lpmf(self, y, x, alpha, beta, phi):
        self._function('neg_binomial_2_log_glm_lpmf', [y, x, alpha, beta, phi])          

    def neg_binomial_2_log_glm_lupmf(self, y, x, alpha, beta, phi):
        self._function('neg_binomial_2_log_glm_lupmf', [y, x, alpha, beta, phi])          

    def neg_binomial_2_log_glm_with_left_and_right_censoring(self, x, alpha, beta, phi, event_left, event_right):
        self._function('neg_binomial_2_log_glm_with_left_and_right_censoring', [x, alpha, beta, phi, event_left, event_right])          

    def neg_binomial_2_log_glm_with_left_and_right_censoring_lpmf(self, y, x, alpha, beta, phi, event_left, event_right):
        self._function('neg_binomial_2_log_glm_with_left_and_right_censoring_lpmf', [y, x, alpha, beta, phi, event_left, event_right])          

    def neg_binomial_2_log_glm_with_left_and_right_censoring_rng(self, x, alpha, beta, phi, event_left, event_right):
        self._function('neg_binomial_2_log_glm_with_left_and_right_censoring_rng', [x, alpha, beta, phi, event_left, event_right])          

    def neg_binomial_2_log_glm_with_left_censoring(self, x, alpha, beta, phi, event):
        self._function('neg_binomial_2_log_glm_with_left_censoring', [x, alpha, beta, phi, event])          

    def neg_binomial_2_log_glm_with_left_censoring_lpmf(self, y, x, alpha, beta, phi, event):
        self._function('neg_binomial_2_log_glm_with_left_censoring_lpmf', [y, x, alpha, beta, phi, event])          

    def neg_binomial_2_log_glm_with_left_censoring_rng(self, y, x, alpha, beta, phi, event):
        self._function('neg_binomial_2_log_glm_with_left_censoring_rng', [y, x, alpha, beta, phi, event])          

    def neg_binomial_2_log_glm_with_right_censoring(self, x, alpha, beta, phi):
        self._function('neg_binomial_2_log_glm_with_right_censoring', [x, alpha, beta, phi])          

    def neg_binomial_2_log_glm_with_right_censoring_lpmf(self, y, x, alpha, beta, phi, event):
        self._function('neg_binomial_2_log_glm_with_right_censoring_lpmf', [y, x, alpha, beta, phi, event])          

    def neg_binomial_2_log_glm_with_right_censoring_rng(self, y, x, alpha, beta, phi, event):
        self._function('neg_binomial_2_log_glm_with_right_censoring_rng', [y, x, alpha, beta, phi, event])          

    def neg_binomial_2_log_lpmf(self, n, eta, phi):
        self._function('neg_binomial_2_log_lpmf', [n, eta, phi])          

    def neg_binomial_2_log_lupmf(self, n, eta, phi):
        self._function('neg_binomial_2_log_lupmf', [n, eta, phi])          

    def neg_binomial_2_log_rng(self, eta, phi):
        self._function('neg_binomial_2_log_rng', [eta, phi])          

    def neg_binomial_2_log_with_left_and_right_censoring(self, eta, phi, event_left, event_right):
        self._function('neg_binomial_2_log_with_left_and_right_censoring', [eta, phi, event_left, event_right])          

    def neg_binomial_2_log_with_left_and_right_censoring_lpmf(self, n, eta, phi, event_left, event_right):
        self._function('neg_binomial_2_log_with_left_and_right_censoring_lpmf', [n, eta, phi, event_left, event_right])          

    def neg_binomial_2_log_with_left_and_right_censoring_rng(self, eta, phi, event_left, event_right):
        self._function('neg_binomial_2_log_with_left_and_right_censoring_rng', [eta, phi, event_left, event_right])          

    def neg_binomial_2_log_with_left_censoring(self, eta, phi, event):
        self._function('neg_binomial_2_log_with_left_censoring', [eta, phi, event])          

    def neg_binomial_2_log_with_left_censoring_lpmf(self, n, eta, phi, event):
        self._function('neg_binomial_2_log_with_left_censoring_lpmf', [n, eta, phi, event])          

    def neg_binomial_2_log_with_left_censoring_rng(self, n, eta, phi, event):
        self._function('neg_binomial_2_log_with_left_censoring_rng', [n, eta, phi, event])          

    def neg_binomial_2_log_with_right_censoring(self, eta, phi):
        self._function('neg_binomial_2_log_with_right_censoring', [eta, phi])          

    def neg_binomial_2_log_with_right_censoring_lpmf(self, n, eta, phi, event):
        self._function('neg_binomial_2_log_with_right_censoring_lpmf', [n, eta, phi, event])          

    def neg_binomial_2_log_with_right_censoring_rng(self, n, eta, phi, event):
        self._function('neg_binomial_2_log_with_right_censoring_rng', [n, eta, phi, event])          

    def neg_binomial_2_lpmf(self, n, mu, phi):
        self._function('neg_binomial_2_lpmf', [n, mu, phi])          

    def neg_binomial_2_lupmf(self, n, mu, phi):
        self._function('neg_binomial_2_lupmf', [n, mu, phi])          

    def neg_binomial_2_rng(self, mu, phi):
        self._function('neg_binomial_2_rng', [mu, phi])          

    def neg_binomial_2_with_left_and_right_censoring(self, mu, phi, event_left, event_right):
        self._function('neg_binomial_2_with_left_and_right_censoring', [mu, phi, event_left, event_right])          

    def neg_binomial_2_with_left_and_right_censoring_lpmf(self, n, mu, phi, event_left, event_right):
        self._function('neg_binomial_2_with_left_and_right_censoring_lpmf', [n, mu, phi, event_left, event_right])          

    def neg_binomial_2_with_left_and_right_censoring_rng(self, mu, phi, event_left, event_right):
        self._function('neg_binomial_2_with_left_and_right_censoring_rng', [mu, phi, event_left, event_right])          

    def neg_binomial_2_with_left_censoring(self, mu, phi, event):
        self._function('neg_binomial_2_with_left_censoring', [mu, phi, event])          

    def neg_binomial_2_with_left_censoring_lpmf(self, n, mu, phi, event):
        self._function('neg_binomial_2_with_left_censoring_lpmf', [n, mu, phi, event])          

    def neg_binomial_2_with_left_censoring_rng(self, n, mu, phi, event):
        self._function('neg_binomial_2_with_left_censoring_rng', [n, mu, phi, event])          

    def neg_binomial_2_with_right_censoring(self, mu, phi):
        self._function('neg_binomial_2_with_right_censoring', [mu, phi])          

    def neg_binomial_2_with_right_censoring_lpmf(self, n, mu, phi, event):
        self._function('neg_binomial_2_with_right_censoring_lpmf', [n, mu, phi, event])          

    def neg_binomial_2_with_right_censoring_rng(self, n, mu, phi, event):
        self._function('neg_binomial_2_with_right_censoring_rng', [n, mu, phi, event])          

    def neg_binomial_cdf(self, n, alpha, beta):
        self._function('neg_binomial_cdf', [n, alpha, beta])          

    def neg_binomial_lccdf(self, n, alpha, beta):
        self._function('neg_binomial_lccdf', [n, alpha, beta])          

    def neg_binomial_lcdf(self, n, alpha, beta):
        self._function('neg_binomial_lcdf', [n, alpha, beta])          

    def neg_binomial_lpmf(self, n, alpha, beta):
        self._function('neg_binomial_lpmf', [n, alpha, beta])          

    def neg_binomial_lupmf(self, n, alpha, beta):
        self._function('neg_binomial_lupmf', [n, alpha, beta])          

    def neg_binomial_rng(self, alpha, beta):
        self._function('neg_binomial_rng', [alpha, beta])          

    def neg_binomial_with_left_and_right_censoring(self, alpha, beta, event_left, event_right):
        self._function('neg_binomial_with_left_and_right_censoring', [alpha, beta, event_left, event_right])          

    def neg_binomial_with_left_and_right_censoring_lpmf(self, n, alpha, beta, event_left, event_right):
        self._function('neg_binomial_with_left_and_right_censoring_lpmf', [n, alpha, beta, event_left, event_right])          

    def neg_binomial_with_left_and_right_censoring_rng(self, alpha, beta, event_left, event_right):
        self._function('neg_binomial_with_left_and_right_censoring_rng', [alpha, beta, event_left, event_right])          

    def neg_binomial_with_left_censoring(self, alpha, beta, event):
        self._function('neg_binomial_with_left_censoring', [alpha, beta, event])          

    def neg_binomial_with_left_censoring_lpmf(self, n, alpha, beta, event):
        self._function('neg_binomial_with_left_censoring_lpmf', [n, alpha, beta, event])          

    def neg_binomial_with_left_censoring_rng(self, n, alpha, beta, event):
        self._function('neg_binomial_with_left_censoring_rng', [n, alpha, beta, event])          

    def neg_binomial_with_right_censoring(self, alpha, beta):
        self._function('neg_binomial_with_right_censoring', [alpha, beta])          

    def neg_binomial_with_right_censoring_lpmf(self, n, alpha, beta, event):
        self._function('neg_binomial_with_right_censoring_lpmf', [n, alpha, beta, event])          

    def neg_binomial_with_right_censoring_rng(self, n, alpha, beta, event):
        self._function('neg_binomial_with_right_censoring_rng', [n, alpha, beta, event])          

    def norm(self, z):
        self._function('norm', [z])          

    def norm1(self, x):
        self._function('norm1', [x])          

    def norm2(self, x):
        self._function('norm2', [x])          

    def normal(self, mu, sigma):
        self._function('normal', [mu, sigma])          

    def normal_cdf(self, y, mu, sigma):
        self._function('normal_cdf', [y, mu, sigma])          

    def normal_id_glm(self, x, alpha, beta, sigma):
        self._function('normal_id_glm', [x, alpha, beta, sigma])          

    def normal_id_glm_lpdf(self, y, x, alpha, beta, sigma):
        self._function('normal_id_glm_lpdf', [y, x, alpha, beta, sigma])          

    def normal_id_glm_lupdf(self, y, x, alpha, beta, sigma):
        self._function('normal_id_glm_lupdf', [y, x, alpha, beta, sigma])          

    def normal_id_glm_with_left_and_right_censoring(self, x, alpha, beta, sigma, event_left, event_right):
        self._function('normal_id_glm_with_left_and_right_censoring', [x, alpha, beta, sigma, event_left, event_right])          

    def normal_id_glm_with_left_and_right_censoring_lpdf(self, y, x, alpha, beta, sigma, event_left, event_right):
        self._function('normal_id_glm_with_left_and_right_censoring_lpdf', [y, x, alpha, beta, sigma, event_left, event_right])          

    def normal_id_glm_with_left_and_right_censoring_rng(self, x, alpha, beta, sigma, event_left, event_right):
        self._function('normal_id_glm_with_left_and_right_censoring_rng', [x, alpha, beta, sigma, event_left, event_right])          

    def normal_id_glm_with_left_censoring(self, x, alpha, beta, sigma, event):
        self._function('normal_id_glm_with_left_censoring', [x, alpha, beta, sigma, event])          

    def normal_id_glm_with_left_censoring_lpdf(self, y, x, alpha, beta, sigma, event):
        self._function('normal_id_glm_with_left_censoring_lpdf', [y, x, alpha, beta, sigma, event])          

    def normal_id_glm_with_left_censoring_rng(self, y, x, alpha, beta, sigma, event):
        self._function('normal_id_glm_with_left_censoring_rng', [y, x, alpha, beta, sigma, event])          

    def normal_id_glm_with_right_censoring(self, x, alpha, beta, sigma):
        self._function('normal_id_glm_with_right_censoring', [x, alpha, beta, sigma])          

    def normal_id_glm_with_right_censoring_lpdf(self, y, x, alpha, beta, sigma, event):
        self._function('normal_id_glm_with_right_censoring_lpdf', [y, x, alpha, beta, sigma, event])          

    def normal_id_glm_with_right_censoring_rng(self, y, x, alpha, beta, sigma, event):
        self._function('normal_id_glm_with_right_censoring_rng', [y, x, alpha, beta, sigma, event])          

    def normal_lccdf(self, y, mu, sigma):
        self._function('normal_lccdf', [y, mu, sigma])          

    def normal_lcdf(self, y, mu, sigma):
        self._function('normal_lcdf', [y, mu, sigma])          

    def normal_lpdf(self, y, mu, sigma):
        self._function('normal_lpdf', [y, mu, sigma])          

    def normal_lupdf(self, y, mu, sigma):
        self._function('normal_lupdf', [y, mu, sigma])          

    def normal_rng(self, mu, sigma):
        self._function('normal_rng', [mu, sigma])          

    def normal_with_left_and_right_censoring(self, mu, sigma, event_left, event_right):
        self._function('normal_with_left_and_right_censoring', [mu, sigma, event_left, event_right])          

    def normal_with_left_and_right_censoring_lpdf(self, y, mu, sigma, event_left, event_right):
        self._function('normal_with_left_and_right_censoring_lpdf', [y, mu, sigma, event_left, event_right])          

    def normal_with_left_and_right_censoring_rng(self, mu, sigma, event_left, event_right):
        self._function('normal_with_left_and_right_censoring_rng', [mu, sigma, event_left, event_right])          

    def normal_with_left_censoring(self, mu, sigma, event):
        self._function('normal_with_left_censoring', [mu, sigma, event])          

    def normal_with_left_censoring_lpdf(self, y, mu, sigma, event):
        self._function('normal_with_left_censoring_lpdf', [y, mu, sigma, event])          

    def normal_with_left_censoring_rng(self, y, mu, sigma, event):
        self._function('normal_with_left_censoring_rng', [y, mu, sigma, event])          

    def normal_with_right_censoring(self, mu, sigma):
        self._function('normal_with_right_censoring', [mu, sigma])          

    def normal_with_right_censoring_lpdf(self, y, mu, sigma, event):
        self._function('normal_with_right_censoring_lpdf', [y, mu, sigma, event])          

    def normal_with_right_censoring_rng(self, y, mu, sigma, event):
        self._function('normal_with_right_censoring_rng', [y, mu, sigma, event])          

    def num_elements(self, x):
        self._function('num_elements', [x])          

    def one_hot_array(self, n, k):
        self._function('one_hot_array', [n, k])          

    def one_hot_int_array(self, n, k):
        self._function('one_hot_int_array', [n, k])          

    def one_hot_row_vector(self, n, k):
        self._function('one_hot_row_vector', [n, k])          

    def one_hot_vector(self, K, k):
        self._function('one_hot_vector', [K, k])          

    def ones_array(self, n):
        self._function('ones_array', [n])          

    def ones_int_array(self, n):
        self._function('ones_int_array', [n])          

    def ones_row_vector(self, n):
        self._function('ones_row_vector', [n])          

    def ones_vector(self, n):
        self._function('ones_vector', [n])          

    def ordered_logistic(self, eta, c):
        self._function('ordered_logistic', [eta, c])          

    def ordered_logistic_glm(self, x, beta, c):
        self._function('ordered_logistic_glm', [x, beta, c])          

    def ordered_logistic_glm_lpmf(self, y, x, beta, c):
        self._function('ordered_logistic_glm_lpmf', [y, x, beta, c])          

    def ordered_logistic_glm_lupmf(self, y, x, beta, c):
        self._function('ordered_logistic_glm_lupmf', [y, x, beta, c])          

    def ordered_logistic_glm_with_left_and_right_censoring(self, x, beta, c, event_left, event_right):
        self._function('ordered_logistic_glm_with_left_and_right_censoring', [x, beta, c, event_left, event_right])          

    def ordered_logistic_glm_with_left_and_right_censoring_lpmf(self, y, x, beta, c, event_left, event_right):
        self._function('ordered_logistic_glm_with_left_and_right_censoring_lpmf', [y, x, beta, c, event_left, event_right])          

    def ordered_logistic_glm_with_left_and_right_censoring_rng(self, x, beta, c, event_left, event_right):
        self._function('ordered_logistic_glm_with_left_and_right_censoring_rng', [x, beta, c, event_left, event_right])          

    def ordered_logistic_glm_with_left_censoring(self, x, beta, c, event):
        self._function('ordered_logistic_glm_with_left_censoring', [x, beta, c, event])          

    def ordered_logistic_glm_with_left_censoring_lpmf(self, y, x, beta, c, event):
        self._function('ordered_logistic_glm_with_left_censoring_lpmf', [y, x, beta, c, event])          

    def ordered_logistic_glm_with_left_censoring_rng(self, y, x, beta, c, event):
        self._function('ordered_logistic_glm_with_left_censoring_rng', [y, x, beta, c, event])          

    def ordered_logistic_glm_with_right_censoring(self, x, beta, c):
        self._function('ordered_logistic_glm_with_right_censoring', [x, beta, c])          

    def ordered_logistic_glm_with_right_censoring_lpmf(self, y, x, beta, c, event):
        self._function('ordered_logistic_glm_with_right_censoring_lpmf', [y, x, beta, c, event])          

    def ordered_logistic_glm_with_right_censoring_rng(self, y, x, beta, c, event):
        self._function('ordered_logistic_glm_with_right_censoring_rng', [y, x, beta, c, event])          

    def ordered_logistic_lpmf(self, k, eta, c):
        self._function('ordered_logistic_lpmf', [k, eta, c])          

    def ordered_logistic_lupmf(self, k, eta, c):
        self._function('ordered_logistic_lupmf', [k, eta, c])          

    def ordered_logistic_rng(self, eta, c):
        self._function('ordered_logistic_rng', [eta, c])          

    def ordered_logistic_with_left_and_right_censoring(self, eta, c, event_left, event_right):
        self._function('ordered_logistic_with_left_and_right_censoring', [eta, c, event_left, event_right])          

    def ordered_logistic_with_left_and_right_censoring_lpmf(self, k, eta, c, event_left, event_right):
        self._function('ordered_logistic_with_left_and_right_censoring_lpmf', [k, eta, c, event_left, event_right])          

    def ordered_logistic_with_left_and_right_censoring_rng(self, eta, c, event_left, event_right):
        self._function('ordered_logistic_with_left_and_right_censoring_rng', [eta, c, event_left, event_right])          

    def ordered_logistic_with_left_censoring(self, eta, c, event):
        self._function('ordered_logistic_with_left_censoring', [eta, c, event])          

    def ordered_logistic_with_left_censoring_lpmf(self, k, eta, c, event):
        self._function('ordered_logistic_with_left_censoring_lpmf', [k, eta, c, event])          

    def ordered_logistic_with_left_censoring_rng(self, k, eta, c, event):
        self._function('ordered_logistic_with_left_censoring_rng', [k, eta, c, event])          

    def ordered_logistic_with_right_censoring(self, eta, c):
        self._function('ordered_logistic_with_right_censoring', [eta, c])          

    def ordered_logistic_with_right_censoring_lpmf(self, k, eta, c, event):
        self._function('ordered_logistic_with_right_censoring_lpmf', [k, eta, c, event])          

    def ordered_logistic_with_right_censoring_rng(self, k, eta, c, event):
        self._function('ordered_logistic_with_right_censoring_rng', [k, eta, c, event])          

    def ordered_probit(self, eta, c):
        self._function('ordered_probit', [eta, c])          

    def ordered_probit_lpmf(self, k, eta, c):
        self._function('ordered_probit_lpmf', [k, eta, c])          

    def ordered_probit_lupmf(self, k, eta, c):
        self._function('ordered_probit_lupmf', [k, eta, c])          

    def ordered_probit_rng(self, eta, c):
        self._function('ordered_probit_rng', [eta, c])          

    def ordered_probit_with_left_and_right_censoring(self, eta, c, event_left, event_right):
        self._function('ordered_probit_with_left_and_right_censoring', [eta, c, event_left, event_right])          

    def ordered_probit_with_left_and_right_censoring_lpmf(self, k, eta, c, event_left, event_right):
        self._function('ordered_probit_with_left_and_right_censoring_lpmf', [k, eta, c, event_left, event_right])          

    def ordered_probit_with_left_and_right_censoring_rng(self, eta, c, event_left, event_right):
        self._function('ordered_probit_with_left_and_right_censoring_rng', [eta, c, event_left, event_right])          

    def ordered_probit_with_left_censoring(self, eta, c, event):
        self._function('ordered_probit_with_left_censoring', [eta, c, event])          

    def ordered_probit_with_left_censoring_lpmf(self, k, eta, c, event):
        self._function('ordered_probit_with_left_censoring_lpmf', [k, eta, c, event])          

    def ordered_probit_with_left_censoring_rng(self, k, eta, c, event):
        self._function('ordered_probit_with_left_censoring_rng', [k, eta, c, event])          

    def ordered_probit_with_right_censoring(self, eta, c):
        self._function('ordered_probit_with_right_censoring', [eta, c])          

    def ordered_probit_with_right_censoring_lpmf(self, k, eta, c, event):
        self._function('ordered_probit_with_right_censoring_lpmf', [k, eta, c, event])          

    def ordered_probit_with_right_censoring_rng(self, k, eta, c, event):
        self._function('ordered_probit_with_right_censoring_rng', [k, eta, c, event])          

    def owens_t(self, h, a):
        self._function('owens_t', [h, a])          

    def pareto(self, y_min, alpha):
        self._function('pareto', [y_min, alpha])          

    def pareto_cdf(self, y, y_min, alpha):
        self._function('pareto_cdf', [y, y_min, alpha])          

    def pareto_lccdf(self, y, y_min, alpha):
        self._function('pareto_lccdf', [y, y_min, alpha])          

    def pareto_lcdf(self, y, y_min, alpha):
        self._function('pareto_lcdf', [y, y_min, alpha])          

    def pareto_lpdf(self, y, y_min, alpha):
        self._function('pareto_lpdf', [y, y_min, alpha])          

    def pareto_lupdf(self, y, y_min, alpha):
        self._function('pareto_lupdf', [y, y_min, alpha])          

    def pareto_rng(self, y_min, alpha):
        self._function('pareto_rng', [y_min, alpha])          

    def pareto_type_2(self, mu, lambda_, alpha):
        self._function('pareto_type_2', [mu, lambda_, alpha])          

    def pareto_type_2_cdf(self, y, mu, lambda_, alpha):
        self._function('pareto_type_2_cdf', [y, mu, lambda_, alpha])          

    def pareto_type_2_lccdf(self, y, mu, lambda_, alpha):
        self._function('pareto_type_2_lccdf', [y, mu, lambda_, alpha])          

    def pareto_type_2_lcdf(self, y, mu, lambda_, alpha):
        self._function('pareto_type_2_lcdf', [y, mu, lambda_, alpha])          

    def pareto_type_2_lpdf(self, y, mu, lambda_, alpha):
        self._function('pareto_type_2_lpdf', [y, mu, lambda_, alpha])          

    def pareto_type_2_lupdf(self, y, mu, lambda_, alpha):
        self._function('pareto_type_2_lupdf', [y, mu, lambda_, alpha])          

    def pareto_type_2_rng(self, mu, lambda_, alpha):
        self._function('pareto_type_2_rng', [mu, lambda_, alpha])          

    def pareto_type_2_with_left_and_right_censoring(self, mu, lambda_, alpha, event_left, event_right):
        self._function('pareto_type_2_with_left_and_right_censoring', [mu, lambda_, alpha, event_left, event_right])          

    def pareto_type_2_with_left_and_right_censoring_lpdf(self, y, mu, lambda_, alpha, event_left, event_right):
        self._function('pareto_type_2_with_left_and_right_censoring_lpdf', [y, mu, lambda_, alpha, event_left, event_right])          

    def pareto_type_2_with_left_and_right_censoring_rng(self, mu, lambda_, alpha, event_left, event_right):
        self._function('pareto_type_2_with_left_and_right_censoring_rng', [mu, lambda_, alpha, event_left, event_right])          

    def pareto_type_2_with_left_censoring(self, mu, lambda_, alpha, event):
        self._function('pareto_type_2_with_left_censoring', [mu, lambda_, alpha, event])          

    def pareto_type_2_with_left_censoring_lpdf(self, y, mu, lambda_, alpha, event):
        self._function('pareto_type_2_with_left_censoring_lpdf', [y, mu, lambda_, alpha, event])          

    def pareto_type_2_with_left_censoring_rng(self, y, mu, lambda_, alpha, event):
        self._function('pareto_type_2_with_left_censoring_rng', [y, mu, lambda_, alpha, event])          

    def pareto_type_2_with_right_censoring(self, mu, lambda_, alpha):
        self._function('pareto_type_2_with_right_censoring', [mu, lambda_, alpha])          

    def pareto_type_2_with_right_censoring_lpdf(self, y, mu, lambda_, alpha, event):
        self._function('pareto_type_2_with_right_censoring_lpdf', [y, mu, lambda_, alpha, event])          

    def pareto_type_2_with_right_censoring_rng(self, y, mu, lambda_, alpha, event):
        self._function('pareto_type_2_with_right_censoring_rng', [y, mu, lambda_, alpha, event])          

    def pareto_with_left_and_right_censoring(self, y_min, alpha, event_left, event_right):
        self._function('pareto_with_left_and_right_censoring', [y_min, alpha, event_left, event_right])          

    def pareto_with_left_and_right_censoring_lpdf(self, y, y_min, alpha, event_left, event_right):
        self._function('pareto_with_left_and_right_censoring_lpdf', [y, y_min, alpha, event_left, event_right])          

    def pareto_with_left_and_right_censoring_rng(self, y_min, alpha, event_left, event_right):
        self._function('pareto_with_left_and_right_censoring_rng', [y_min, alpha, event_left, event_right])          

    def pareto_with_left_censoring(self, y_min, alpha, event):
        self._function('pareto_with_left_censoring', [y_min, alpha, event])          

    def pareto_with_left_censoring_lpdf(self, y, y_min, alpha, event):
        self._function('pareto_with_left_censoring_lpdf', [y, y_min, alpha, event])          

    def pareto_with_left_censoring_rng(self, y, y_min, alpha, event):
        self._function('pareto_with_left_censoring_rng', [y, y_min, alpha, event])          

    def pareto_with_right_censoring(self, y_min, alpha):
        self._function('pareto_with_right_censoring', [y_min, alpha])          

    def pareto_with_right_censoring_lpdf(self, y, y_min, alpha, event):
        self._function('pareto_with_right_censoring_lpdf', [y, y_min, alpha, event])          

    def pareto_with_right_censoring_rng(self, y, y_min, alpha, event):
        self._function('pareto_with_right_censoring_rng', [y, y_min, alpha, event])          

    def Phi(self, x):
        self._function('Phi', [x])          

    def Phi_approx(self, x):
        self._function('Phi_approx', [x])          

    def poisson(self, lambda_):
        self._function('poisson', [lambda_])          

    def poisson_cdf(self, n, lambda_):
        self._function('poisson_cdf', [n, lambda_])          

    def poisson_lccdf(self, n, lambda_):
        self._function('poisson_lccdf', [n, lambda_])          

    def poisson_lcdf(self, n, lambda_):
        self._function('poisson_lcdf', [n, lambda_])          

    def poisson_log(self, alpha):
        self._function('poisson_log', [alpha])          

    def poisson_log_glm(self, x, alpha, beta):
        self._function('poisson_log_glm', [x, alpha, beta])          

    def poisson_log_glm_lpmf(self, y, x, alpha, beta):
        self._function('poisson_log_glm_lpmf', [y, x, alpha, beta])          

    def poisson_log_glm_lupmf(self, y, x, alpha, beta):
        self._function('poisson_log_glm_lupmf', [y, x, alpha, beta])          

    def poisson_log_glm_with_left_and_right_censoring(self, x, alpha, beta, event_left, event_right):
        self._function('poisson_log_glm_with_left_and_right_censoring', [x, alpha, beta, event_left, event_right])          

    def poisson_log_glm_with_left_and_right_censoring_lpmf(self, y, x, alpha, beta, event_left, event_right):
        self._function('poisson_log_glm_with_left_and_right_censoring_lpmf', [y, x, alpha, beta, event_left, event_right])          

    def poisson_log_glm_with_left_and_right_censoring_rng(self, x, alpha, beta, event_left, event_right):
        self._function('poisson_log_glm_with_left_and_right_censoring_rng', [x, alpha, beta, event_left, event_right])          

    def poisson_log_glm_with_left_censoring(self, x, alpha, beta, event):
        self._function('poisson_log_glm_with_left_censoring', [x, alpha, beta, event])          

    def poisson_log_glm_with_left_censoring_lpmf(self, y, x, alpha, beta, event):
        self._function('poisson_log_glm_with_left_censoring_lpmf', [y, x, alpha, beta, event])          

    def poisson_log_glm_with_left_censoring_rng(self, y, x, alpha, beta, event):
        self._function('poisson_log_glm_with_left_censoring_rng', [y, x, alpha, beta, event])          

    def poisson_log_glm_with_right_censoring(self, x, alpha, beta):
        self._function('poisson_log_glm_with_right_censoring', [x, alpha, beta])          

    def poisson_log_glm_with_right_censoring_lpmf(self, y, x, alpha, beta, event):
        self._function('poisson_log_glm_with_right_censoring_lpmf', [y, x, alpha, beta, event])          

    def poisson_log_glm_with_right_censoring_rng(self, y, x, alpha, beta, event):
        self._function('poisson_log_glm_with_right_censoring_rng', [y, x, alpha, beta, event])          

    def poisson_log_lpmf(self, n, alpha):
        self._function('poisson_log_lpmf', [n, alpha])          

    def poisson_log_lupmf(self, n, alpha):
        self._function('poisson_log_lupmf', [n, alpha])          

    def poisson_log_rng(self, alpha):
        self._function('poisson_log_rng', [alpha])          

    def poisson_log_with_left_and_right_censoring(self, alpha, event_left, event_right):
        self._function('poisson_log_with_left_and_right_censoring', [alpha, event_left, event_right])          

    def poisson_log_with_left_and_right_censoring_lpmf(self, n, alpha, event_left, event_right):
        self._function('poisson_log_with_left_and_right_censoring_lpmf', [n, alpha, event_left, event_right])          

    def poisson_log_with_left_and_right_censoring_rng(self, alpha, event_left, event_right):
        self._function('poisson_log_with_left_and_right_censoring_rng', [alpha, event_left, event_right])          

    def poisson_log_with_left_censoring(self, alpha, event):
        self._function('poisson_log_with_left_censoring', [alpha, event])          

    def poisson_log_with_left_censoring_lpmf(self, n, alpha, event):
        self._function('poisson_log_with_left_censoring_lpmf', [n, alpha, event])          

    def poisson_log_with_left_censoring_rng(self, n, alpha, event):
        self._function('poisson_log_with_left_censoring_rng', [n, alpha, event])          

    def poisson_log_with_right_censoring(self, alpha):
        self._function('poisson_log_with_right_censoring', [alpha])          

    def poisson_log_with_right_censoring_lpmf(self, n, alpha, event):
        self._function('poisson_log_with_right_censoring_lpmf', [n, alpha, event])          

    def poisson_log_with_right_censoring_rng(self, n, alpha, event):
        self._function('poisson_log_with_right_censoring_rng', [n, alpha, event])          

    def poisson_lpmf(self, n, lambda_):
        self._function('poisson_lpmf', [n, lambda_])          

    def poisson_lupmf(self, n, lambda_):
        self._function('poisson_lupmf', [n, lambda_])          

    def poisson_rng(self, lambda_):
        self._function('poisson_rng', [lambda_])          

    def poisson_with_left_and_right_censoring(self, lambda_, event_left, event_right):
        self._function('poisson_with_left_and_right_censoring', [lambda_, event_left, event_right])          

    def poisson_with_left_and_right_censoring_lpmf(self, n, lambda_, event_left, event_right):
        self._function('poisson_with_left_and_right_censoring_lpmf', [n, lambda_, event_left, event_right])          

    def poisson_with_left_and_right_censoring_rng(self, lambda_, event_left, event_right):
        self._function('poisson_with_left_and_right_censoring_rng', [lambda_, event_left, event_right])          

    def poisson_with_left_censoring(self, lambda_, event):
        self._function('poisson_with_left_censoring', [lambda_, event])          

    def poisson_with_left_censoring_lpmf(self, n, lambda_, event):
        self._function('poisson_with_left_censoring_lpmf', [n, lambda_, event])          

    def poisson_with_left_censoring_rng(self, n, lambda_, event):
        self._function('poisson_with_left_censoring_rng', [n, lambda_, event])          

    def poisson_with_right_censoring(self, lambda_):
        self._function('poisson_with_right_censoring', [lambda_])          

    def poisson_with_right_censoring_lpmf(self, n, lambda_, event):
        self._function('poisson_with_right_censoring_lpmf', [n, lambda_, event])          

    def poisson_with_right_censoring_rng(self, n, lambda_, event):
        self._function('poisson_with_right_censoring_rng', [n, lambda_, event])          

    def polar(self, r, theta):
        self._function('polar', [r, theta])          

    def pow(self, x, y):
        self._function('pow', [x, y])          

    def prod(self, x):
        self._function('prod', [x])          

    def proj(self, z):
        self._function('proj', [z])          

    def qr(self, A):
        self._function('qr', [A])          

    def qr_Q(self, A):
        self._function('qr_Q', [A])          

    def qr_R(self, A):
        self._function('qr_R', [A])          

    def qr_thin(self, A):
        self._function('qr_thin', [A])          

    def qr_thin_Q(self, A):
        self._function('qr_thin_Q', [A])          

    def qr_thin_R(self, A):
        self._function('qr_thin_R', [A])          

    def quad_form(self, A, B):
        self._function('quad_form', [A, B])          

    def quad_form_diag(self, m, v):
        self._function('quad_form_diag', [m, v])          

    def quad_form_sym(self, A, B):
        self._function('quad_form_sym', [A, B])          

    def quantile(self, x, p):
        self._function('quantile', [x, p])          

    def rank(self, v, s):
        self._function('rank', [v, s])          

    def rayleigh(self, sigma):
        self._function('rayleigh', [sigma])          

    def rayleigh_cdf(self, y, sigma):
        self._function('rayleigh_cdf', [y, sigma])          

    def rayleigh_lccdf(self, y, sigma):
        self._function('rayleigh_lccdf', [y, sigma])          

    def rayleigh_lcdf(self, y, sigma):
        self._function('rayleigh_lcdf', [y, sigma])          

    def rayleigh_lpdf(self, y, sigma):
        self._function('rayleigh_lpdf', [y, sigma])          

    def rayleigh_lupdf(self, y, sigma):
        self._function('rayleigh_lupdf', [y, sigma])          

    def rayleigh_rng(self, sigma):
        self._function('rayleigh_rng', [sigma])          

    def rayleigh_with_left_and_right_censoring(self, sigma, event_left, event_right):
        self._function('rayleigh_with_left_and_right_censoring', [sigma, event_left, event_right])          

    def rayleigh_with_left_and_right_censoring_lpdf(self, y, sigma, event_left, event_right):
        self._function('rayleigh_with_left_and_right_censoring_lpdf', [y, sigma, event_left, event_right])          

    def rayleigh_with_left_and_right_censoring_rng(self, sigma, event_left, event_right):
        self._function('rayleigh_with_left_and_right_censoring_rng', [sigma, event_left, event_right])          

    def rayleigh_with_left_censoring(self, sigma, event):
        self._function('rayleigh_with_left_censoring', [sigma, event])          

    def rayleigh_with_left_censoring_lpdf(self, y, sigma, event):
        self._function('rayleigh_with_left_censoring_lpdf', [y, sigma, event])          

    def rayleigh_with_left_censoring_rng(self, y, sigma, event):
        self._function('rayleigh_with_left_censoring_rng', [y, sigma, event])          

    def rayleigh_with_right_censoring(self, sigma):
        self._function('rayleigh_with_right_censoring', [sigma])          

    def rayleigh_with_right_censoring_lpdf(self, y, sigma, event):
        self._function('rayleigh_with_right_censoring_lpdf', [y, sigma, event])          

    def rayleigh_with_right_censoring_rng(self, y, sigma, event):
        self._function('rayleigh_with_right_censoring_rng', [y, sigma, event])          

    def rep_array(self, x, n):
        self._function('rep_array', [x, n])          

    def rep_matrix(self, z, m, n):
        self._function('rep_matrix', [z, m, n])          

    def rep_row_vector(self, z, n):
        self._function('rep_row_vector', [z, n])          

    def rep_vector(self, z, m):
        self._function('rep_vector', [z, m])          

    def reverse(self, v):
        self._function('reverse', [v])          

    def rising_factorial(self, x, n):
        self._function('rising_factorial', [x, n])          

    def round(self, x):
        self._function('round', [x])          

    def row(self, x, m):
        self._function('row', [x, m])          

    def rows(self, x):
        self._function('rows', [x])          

    def rows_dot_product(self, x, y):
        self._function('rows_dot_product', [x, y])          

    def rows_dot_self(self, x):
        self._function('rows_dot_self', [x])          

    def scale_matrix_exp_multiply(self, t, A, B):
        self._function('scale_matrix_exp_multiply', [t, A, B])          

    def scaled_inv_chi_square(self, nu, sigma):
        self._function('scaled_inv_chi_square', [nu, sigma])          

    def scaled_inv_chi_square_cdf(self, y, nu, sigma):
        self._function('scaled_inv_chi_square_cdf', [y, nu, sigma])          

    def scaled_inv_chi_square_lccdf(self, y, nu, sigma):
        self._function('scaled_inv_chi_square_lccdf', [y, nu, sigma])          

    def scaled_inv_chi_square_lcdf(self, y, nu, sigma):
        self._function('scaled_inv_chi_square_lcdf', [y, nu, sigma])          

    def scaled_inv_chi_square_lpdf(self, y, nu, sigma):
        self._function('scaled_inv_chi_square_lpdf', [y, nu, sigma])          

    def scaled_inv_chi_square_lupdf(self, y, nu, sigma):
        self._function('scaled_inv_chi_square_lupdf', [y, nu, sigma])          

    def scaled_inv_chi_square_rng(self, nu, sigma):
        self._function('scaled_inv_chi_square_rng', [nu, sigma])          

    def scaled_inv_chi_square_with_left_and_right_censoring(self, nu, sigma, event_left, event_right):
        self._function('scaled_inv_chi_square_with_left_and_right_censoring', [nu, sigma, event_left, event_right])          

    def scaled_inv_chi_square_with_left_and_right_censoring_lpdf(self, y, nu, sigma, event_left, event_right):
        self._function('scaled_inv_chi_square_with_left_and_right_censoring_lpdf', [y, nu, sigma, event_left, event_right])          

    def scaled_inv_chi_square_with_left_and_right_censoring_rng(self, nu, sigma, event_left, event_right):
        self._function('scaled_inv_chi_square_with_left_and_right_censoring_rng', [nu, sigma, event_left, event_right])          

    def scaled_inv_chi_square_with_left_censoring(self, nu, sigma, event):
        self._function('scaled_inv_chi_square_with_left_censoring', [nu, sigma, event])          

    def scaled_inv_chi_square_with_left_censoring_lpdf(self, y, nu, sigma, event):
        self._function('scaled_inv_chi_square_with_left_censoring_lpdf', [y, nu, sigma, event])          

    def scaled_inv_chi_square_with_left_censoring_rng(self, y, nu, sigma, event):
        self._function('scaled_inv_chi_square_with_left_censoring_rng', [y, nu, sigma, event])          

    def scaled_inv_chi_square_with_right_censoring(self, nu, sigma):
        self._function('scaled_inv_chi_square_with_right_censoring', [nu, sigma])          

    def scaled_inv_chi_square_with_right_censoring_lpdf(self, y, nu, sigma, event):
        self._function('scaled_inv_chi_square_with_right_censoring_lpdf', [y, nu, sigma, event])          

    def scaled_inv_chi_square_with_right_censoring_rng(self, y, nu, sigma, event):
        self._function('scaled_inv_chi_square_with_right_censoring_rng', [y, nu, sigma, event])          

    def sd(self, x):
        self._function('sd', [x])          

    def segment(self, v, i, n):
        self._function('segment', [v, i, n])          

    def sin(self, z):
        self._function('sin', [z])          

    def singular_values(self, A):
        self._function('singular_values', [A])          

    def sinh(self, z):
        self._function('sinh', [z])          

    def size(self, x):
        self._function('size', [x])          

    def skew_double_exponential(self, mu, sigma, tau):
        self._function('skew_double_exponential', [mu, sigma, tau])          

    def skew_double_exponential_cdf(self, y, mu, sigma, tau):
        self._function('skew_double_exponential_cdf', [y, mu, sigma, tau])          

    def skew_double_exponential_lccdf(self, y, mu, sigma, tau):
        self._function('skew_double_exponential_lccdf', [y, mu, sigma, tau])          

    def skew_double_exponential_lcdf(self, y, mu, sigma, tau):
        self._function('skew_double_exponential_lcdf', [y, mu, sigma, tau])          

    def skew_double_exponential_lpdf(self, y, mu, sigma, tau):
        self._function('skew_double_exponential_lpdf', [y, mu, sigma, tau])          

    def skew_double_exponential_lupdf(self, y, mu, sigma, tau):
        self._function('skew_double_exponential_lupdf', [y, mu, sigma, tau])          

    def skew_double_exponential_rng(self, mu, sigma):
        self._function('skew_double_exponential_rng', [mu, sigma])          

    def skew_double_exponential_with_left_and_right_censoring(self, mu, sigma, tau, event_left, event_right):
        self._function('skew_double_exponential_with_left_and_right_censoring', [mu, sigma, tau, event_left, event_right])          

    def skew_double_exponential_with_left_and_right_censoring_lpdf(self, y, mu, sigma, tau, event_left, event_right):
        self._function('skew_double_exponential_with_left_and_right_censoring_lpdf', [y, mu, sigma, tau, event_left, event_right])          

    def skew_double_exponential_with_left_and_right_censoring_rng(self, mu, sigma, tau, event_left, event_right):
        self._function('skew_double_exponential_with_left_and_right_censoring_rng', [mu, sigma, tau, event_left, event_right])          

    def skew_double_exponential_with_left_censoring(self, mu, sigma, tau, event):
        self._function('skew_double_exponential_with_left_censoring', [mu, sigma, tau, event])          

    def skew_double_exponential_with_left_censoring_lpdf(self, y, mu, sigma, tau, event):
        self._function('skew_double_exponential_with_left_censoring_lpdf', [y, mu, sigma, tau, event])          

    def skew_double_exponential_with_left_censoring_rng(self, y, mu, sigma, tau, event):
        self._function('skew_double_exponential_with_left_censoring_rng', [y, mu, sigma, tau, event])          

    def skew_double_exponential_with_right_censoring(self, mu, sigma, tau):
        self._function('skew_double_exponential_with_right_censoring', [mu, sigma, tau])          

    def skew_double_exponential_with_right_censoring_lpdf(self, y, mu, sigma, tau, event):
        self._function('skew_double_exponential_with_right_censoring_lpdf', [y, mu, sigma, tau, event])          

    def skew_double_exponential_with_right_censoring_rng(self, y, mu, sigma, tau, event):
        self._function('skew_double_exponential_with_right_censoring_rng', [y, mu, sigma, tau, event])          

    def skew_normal(self, xi, omega, alpha):
        self._function('skew_normal', [xi, omega, alpha])          

    def skew_normal_cdf(self, y, xi, omega, alpha):
        self._function('skew_normal_cdf', [y, xi, omega, alpha])          

    def skew_normal_lccdf(self, y, xi, omega, alpha):
        self._function('skew_normal_lccdf', [y, xi, omega, alpha])          

    def skew_normal_lcdf(self, y, xi, omega, alpha):
        self._function('skew_normal_lcdf', [y, xi, omega, alpha])          

    def skew_normal_lpdf(self, y, xi, omega, alpha):
        self._function('skew_normal_lpdf', [y, xi, omega, alpha])          

    def skew_normal_lupdf(self, y, xi, omega, alpha):
        self._function('skew_normal_lupdf', [y, xi, omega, alpha])          

    def skew_normal_rng(self, xi, omega, alpha):
        self._function('skew_normal_rng', [xi, omega, alpha])          

    def skew_normal_with_left_and_right_censoring(self, xi, omega, alpha, event_left, event_right):
        self._function('skew_normal_with_left_and_right_censoring', [xi, omega, alpha, event_left, event_right])          

    def skew_normal_with_left_and_right_censoring_lpdf(self, y, xi, omega, alpha, event_left, event_right):
        self._function('skew_normal_with_left_and_right_censoring_lpdf', [y, xi, omega, alpha, event_left, event_right])          

    def skew_normal_with_left_and_right_censoring_rng(self, xi, omega, alpha, event_left, event_right):
        self._function('skew_normal_with_left_and_right_censoring_rng', [xi, omega, alpha, event_left, event_right])          

    def skew_normal_with_left_censoring(self, xi, omega, alpha, event):
        self._function('skew_normal_with_left_censoring', [xi, omega, alpha, event])          

    def skew_normal_with_left_censoring_lpdf(self, y, xi, omega, alpha, event):
        self._function('skew_normal_with_left_censoring_lpdf', [y, xi, omega, alpha, event])          

    def skew_normal_with_left_censoring_rng(self, y, xi, omega, alpha, event):
        self._function('skew_normal_with_left_censoring_rng', [y, xi, omega, alpha, event])          

    def skew_normal_with_right_censoring(self, xi, omega, alpha):
        self._function('skew_normal_with_right_censoring', [xi, omega, alpha])          

    def skew_normal_with_right_censoring_lpdf(self, y, xi, omega, alpha, event):
        self._function('skew_normal_with_right_censoring_lpdf', [y, xi, omega, alpha, event])          

    def skew_normal_with_right_censoring_rng(self, y, xi, omega, alpha, event):
        self._function('skew_normal_with_right_censoring_rng', [y, xi, omega, alpha, event])          

    def softmax(self, x):
        self._function('softmax', [x])          

    def sort_asc(self, v):
        self._function('sort_asc', [v])          

    def sort_desc(self, v):
        self._function('sort_desc', [v])          

    def sort_indices_asc(self, v):
        self._function('sort_indices_asc', [v])          

    def sort_indices_desc(self, v):
        self._function('sort_indices_desc', [v])          

    def sqrt(self, x):
        self._function('sqrt', [x])          

    def square(self, x):
        self._function('square', [x])          

    def squared_distance(self, x, y):
        self._function('squared_distance', [x, y])          

    def std_normal(self):
        self._function('std_normal', [])          

    def std_normal_cdf(self, y):
        self._function('std_normal_cdf', [y])          

    def std_normal_lccdf(self, y):
        self._function('std_normal_lccdf', [y])          

    def std_normal_lcdf(self, y):
        self._function('std_normal_lcdf', [y])          

    def std_normal_log_qf(self, x):
        self._function('std_normal_log_qf', [x])          

    def std_normal_lpdf(self, y):
        self._function('std_normal_lpdf', [y])          

    def std_normal_lupdf(self, y):
        self._function('std_normal_lupdf', [y])          

    def std_normal_qf(self, x):
        self._function('std_normal_qf', [x])          

    def std_normal_with_left_and_right_censoring(self, event_left, event_right):
        self._function('std_normal_with_left_and_right_censoring', [event_left, event_right])          

    def std_normal_with_left_and_right_censoring_lpdf(self, y, event_left, event_right):
        self._function('std_normal_with_left_and_right_censoring_lpdf', [y, event_left, event_right])          

    def std_normal_with_left_and_right_censoring_rng(self, event_left, event_right):
        self._function('std_normal_with_left_and_right_censoring_rng', [event_left, event_right])          

    def std_normal_with_left_censoring(self, event):
        self._function('std_normal_with_left_censoring', [event])          

    def std_normal_with_left_censoring_lpdf(self, y, event):
        self._function('std_normal_with_left_censoring_lpdf', [y, event])          

    def std_normal_with_left_censoring_rng(self, y, event):
        self._function('std_normal_with_left_censoring_rng', [y, event])          

    def std_normal_with_right_censoring(self):
        self._function('std_normal_with_right_censoring', [])          

    def std_normal_with_right_censoring_lpdf(self, y, event):
        self._function('std_normal_with_right_censoring_lpdf', [y, event])          

    def std_normal_with_right_censoring_rng(self, y, event):
        self._function('std_normal_with_right_censoring_rng', [y, event])          

    def step(self, x):
        self._function('step', [x])          

    def student_t(self, nu, mu, sigma):
        self._function('student_t', [nu, mu, sigma])          

    def student_t_cdf(self, y, nu, mu, sigma):
        self._function('student_t_cdf', [y, nu, mu, sigma])          

    def student_t_lccdf(self, y, nu, mu, sigma):
        self._function('student_t_lccdf', [y, nu, mu, sigma])          

    def student_t_lcdf(self, y, nu, mu, sigma):
        self._function('student_t_lcdf', [y, nu, mu, sigma])          

    def student_t_lpdf(self, y, nu, mu, sigma):
        self._function('student_t_lpdf', [y, nu, mu, sigma])          

    def student_t_lupdf(self, y, nu, mu, sigma):
        self._function('student_t_lupdf', [y, nu, mu, sigma])          

    def student_t_rng(self, nu, mu, sigma):
        self._function('student_t_rng', [nu, mu, sigma])          

    def student_t_with_left_and_right_censoring(self, nu, mu, sigma, event_left, event_right):
        self._function('student_t_with_left_and_right_censoring', [nu, mu, sigma, event_left, event_right])          

    def student_t_with_left_and_right_censoring_lpdf(self, y, nu, mu, sigma, event_left, event_right):
        self._function('student_t_with_left_and_right_censoring_lpdf', [y, nu, mu, sigma, event_left, event_right])          

    def student_t_with_left_and_right_censoring_rng(self, nu, mu, sigma, event_left, event_right):
        self._function('student_t_with_left_and_right_censoring_rng', [nu, mu, sigma, event_left, event_right])          

    def student_t_with_left_censoring(self, nu, mu, sigma, event):
        self._function('student_t_with_left_censoring', [nu, mu, sigma, event])          

    def student_t_with_left_censoring_lpdf(self, y, nu, mu, sigma, event):
        self._function('student_t_with_left_censoring_lpdf', [y, nu, mu, sigma, event])          

    def student_t_with_left_censoring_rng(self, y, nu, mu, sigma, event):
        self._function('student_t_with_left_censoring_rng', [y, nu, mu, sigma, event])          

    def student_t_with_right_censoring(self, nu, mu, sigma):
        self._function('student_t_with_right_censoring', [nu, mu, sigma])          

    def student_t_with_right_censoring_lpdf(self, y, nu, mu, sigma, event):
        self._function('student_t_with_right_censoring_lpdf', [y, nu, mu, sigma, event])          

    def student_t_with_right_censoring_rng(self, y, nu, mu, sigma, event):
        self._function('student_t_with_right_censoring_rng', [y, nu, mu, sigma, event])          

    def sub_col(self, x, i, j, n_rows):
        self._function('sub_col', [x, i, j, n_rows])          

    def sub_row(self, x, i, j, n_cols):
        self._function('sub_row', [x, i, j, n_cols])          

    def sum(self, x):
        self._function('sum', [x])          

    def svd(self, A):
        self._function('svd', [A])          

    def svd_U(self, A):
        self._function('svd_U', [A])          

    def svd_V(self, A):
        self._function('svd_V', [A])          

    def symmetrize_from_lower_tri(self, A):
        self._function('symmetrize_from_lower_tri', [A])          

    def tail(self, v, n):
        self._function('tail', [v, n])          

    def tan(self, z):
        self._function('tan', [z])          

    def tanh(self, z):
        self._function('tanh', [z])          

    def tcrossprod(self, x):
        self._function('tcrossprod', [x])          

    def tgamma(self, x):
        self._function('tgamma', [x])          

    def to_array_1d(self, v):
        self._function('to_array_1d', [v])          

    def to_array_2d(self, m):
        self._function('to_array_2d', [m])          

    def to_complex(self, re):
        self._function('to_complex', [re])          

    def to_int(self, x):
        self._function('to_int', [x])          

    def to_matrix(self, m):
        self._function('to_matrix', [m])          

    def to_row_vector(self, m):
        self._function('to_row_vector', [m])          

    def to_vector(self, m):
        self._function('to_vector', [m])          

    def trace(self, A):
        self._function('trace', [A])          

    def trace_gen_quad_form(self, D, A, B):
        self._function('trace_gen_quad_form', [D, A, B])          

    def trace_quad_form(self, A, B):
        self._function('trace_quad_form', [A, B])          

    def trigamma(self, x):
        self._function('trigamma', [x])          

    def trunc(self, x):
        self._function('trunc', [x])          

    def uniform(self, alpha, beta):
        self._function('uniform', [alpha, beta])          

    def uniform_cdf(self, y, alpha, beta):
        self._function('uniform_cdf', [y, alpha, beta])          

    def uniform_lccdf(self, y, alpha, beta):
        self._function('uniform_lccdf', [y, alpha, beta])          

    def uniform_lcdf(self, y, alpha, beta):
        self._function('uniform_lcdf', [y, alpha, beta])          

    def uniform_lpdf(self, y, alpha, beta):
        self._function('uniform_lpdf', [y, alpha, beta])          

    def uniform_lupdf(self, y, alpha, beta):
        self._function('uniform_lupdf', [y, alpha, beta])          

    def uniform_rng(self, alpha, beta):
        self._function('uniform_rng', [alpha, beta])          

    def uniform_simplex(self, n):
        self._function('uniform_simplex', [n])          

    def uniform_with_left_and_right_censoring(self, alpha, beta, event_left, event_right):
        self._function('uniform_with_left_and_right_censoring', [alpha, beta, event_left, event_right])          

    def uniform_with_left_and_right_censoring_lpdf(self, y, alpha, beta, event_left, event_right):
        self._function('uniform_with_left_and_right_censoring_lpdf', [y, alpha, beta, event_left, event_right])          

    def uniform_with_left_and_right_censoring_rng(self, alpha, beta, event_left, event_right):
        self._function('uniform_with_left_and_right_censoring_rng', [alpha, beta, event_left, event_right])          

    def uniform_with_left_censoring(self, alpha, beta, event):
        self._function('uniform_with_left_censoring', [alpha, beta, event])          

    def uniform_with_left_censoring_lpdf(self, y, alpha, beta, event):
        self._function('uniform_with_left_censoring_lpdf', [y, alpha, beta, event])          

    def uniform_with_left_censoring_rng(self, y, alpha, beta, event):
        self._function('uniform_with_left_censoring_rng', [y, alpha, beta, event])          

    def uniform_with_right_censoring(self, alpha, beta):
        self._function('uniform_with_right_censoring', [alpha, beta])          

    def uniform_with_right_censoring_lpdf(self, y, alpha, beta, event):
        self._function('uniform_with_right_censoring_lpdf', [y, alpha, beta, event])          

    def uniform_with_right_censoring_rng(self, y, alpha, beta, event):
        self._function('uniform_with_right_censoring_rng', [y, alpha, beta, event])          

    def variance(self, x):
        self._function('variance', [x])          

    def von_mises(self, mu, kappa):
        self._function('von_mises', [mu, kappa])          

    def von_mises_cdf(self, y, mu, kappa):
        self._function('von_mises_cdf', [y, mu, kappa])          

    def von_mises_lccdf(self, y, mu, kappa):
        self._function('von_mises_lccdf', [y, mu, kappa])          

    def von_mises_lcdf(self, y, mu, kappa):
        self._function('von_mises_lcdf', [y, mu, kappa])          

    def von_mises_lpdf(self, y, mu, kappa):
        self._function('von_mises_lpdf', [y, mu, kappa])          

    def von_mises_lupdf(self, y, mu, kappa):
        self._function('von_mises_lupdf', [y, mu, kappa])          

    def von_mises_rng(self, mu, kappa):
        self._function('von_mises_rng', [mu, kappa])          

    def von_mises_with_left_and_right_censoring(self, mu, kappa, event_left, event_right):
        self._function('von_mises_with_left_and_right_censoring', [mu, kappa, event_left, event_right])          

    def von_mises_with_left_and_right_censoring_lpdf(self, y, mu, kappa, event_left, event_right):
        self._function('von_mises_with_left_and_right_censoring_lpdf', [y, mu, kappa, event_left, event_right])          

    def von_mises_with_left_and_right_censoring_rng(self, mu, kappa, event_left, event_right):
        self._function('von_mises_with_left_and_right_censoring_rng', [mu, kappa, event_left, event_right])          

    def von_mises_with_left_censoring(self, mu, kappa, event):
        self._function('von_mises_with_left_censoring', [mu, kappa, event])          

    def von_mises_with_left_censoring_lpdf(self, y, mu, kappa, event):
        self._function('von_mises_with_left_censoring_lpdf', [y, mu, kappa, event])          

    def von_mises_with_left_censoring_rng(self, y, mu, kappa, event):
        self._function('von_mises_with_left_censoring_rng', [y, mu, kappa, event])          

    def von_mises_with_right_censoring(self, mu, kappa):
        self._function('von_mises_with_right_censoring', [mu, kappa])          

    def von_mises_with_right_censoring_lpdf(self, y, mu, kappa, event):
        self._function('von_mises_with_right_censoring_lpdf', [y, mu, kappa, event])          

    def von_mises_with_right_censoring_rng(self, y, mu, kappa, event):
        self._function('von_mises_with_right_censoring_rng', [y, mu, kappa, event])          

    def weibull(self, alpha, sigma):
        self._function('weibull', [alpha, sigma])          

    def weibull_cdf(self, y, alpha, sigma):
        self._function('weibull_cdf', [y, alpha, sigma])          

    def weibull_lccdf(self, y, alpha, sigma):
        self._function('weibull_lccdf', [y, alpha, sigma])          

    def weibull_lcdf(self, y, alpha, sigma):
        self._function('weibull_lcdf', [y, alpha, sigma])          

    def weibull_lpdf(self, y, alpha, sigma):
        self._function('weibull_lpdf', [y, alpha, sigma])          

    def weibull_lupdf(self, y, alpha, sigma):
        self._function('weibull_lupdf', [y, alpha, sigma])          

    def weibull_rng(self, alpha, sigma):
        self._function('weibull_rng', [alpha, sigma])          

    def weibull_with_left_and_right_censoring(self, alpha, sigma, event_left, event_right):
        self._function('weibull_with_left_and_right_censoring', [alpha, sigma, event_left, event_right])          

    def weibull_with_left_and_right_censoring_lpdf(self, y, alpha, sigma, event_left, event_right):
        self._function('weibull_with_left_and_right_censoring_lpdf', [y, alpha, sigma, event_left, event_right])          

    def weibull_with_left_and_right_censoring_rng(self, alpha, sigma, event_left, event_right):
        self._function('weibull_with_left_and_right_censoring_rng', [alpha, sigma, event_left, event_right])          

    def weibull_with_left_censoring(self, alpha, sigma, event):
        self._function('weibull_with_left_censoring', [alpha, sigma, event])          

    def weibull_with_left_censoring_lpdf(self, y, alpha, sigma, event):
        self._function('weibull_with_left_censoring_lpdf', [y, alpha, sigma, event])          

    def weibull_with_left_censoring_rng(self, y, alpha, sigma, event):
        self._function('weibull_with_left_censoring_rng', [y, alpha, sigma, event])          

    def weibull_with_right_censoring(self, alpha, sigma):
        self._function('weibull_with_right_censoring', [alpha, sigma])          

    def weibull_with_right_censoring_lpdf(self, y, alpha, sigma, event):
        self._function('weibull_with_right_censoring_lpdf', [y, alpha, sigma, event])          

    def weibull_with_right_censoring_rng(self, y, alpha, sigma, event):
        self._function('weibull_with_right_censoring_rng', [y, alpha, sigma, event])          

    def wiener(self, alpha, tau, beta, delta):
        self._function('wiener', [alpha, tau, beta, delta])          

    def wiener_lpdf(self, y, alpha, tau, beta, delta):
        self._function('wiener_lpdf', [y, alpha, tau, beta, delta])          

    def wiener_lupdf(self, y, alpha, tau, beta, delta):
        self._function('wiener_lupdf', [y, alpha, tau, beta, delta])          

    def wiener_with_left_and_right_censoring(self, alpha, tau, beta, delta, event_left, event_right):
        self._function('wiener_with_left_and_right_censoring', [alpha, tau, beta, delta, event_left, event_right])          

    def wiener_with_left_and_right_censoring_lpdf(self, y, alpha, tau, beta, delta, event_left, event_right):
        self._function('wiener_with_left_and_right_censoring_lpdf', [y, alpha, tau, beta, delta, event_left, event_right])          

    def wiener_with_left_and_right_censoring_rng(self, alpha, tau, beta, delta, event_left, event_right):
        self._function('wiener_with_left_and_right_censoring_rng', [alpha, tau, beta, delta, event_left, event_right])          

    def wiener_with_left_censoring(self, alpha, tau, beta, delta, event):
        self._function('wiener_with_left_censoring', [alpha, tau, beta, delta, event])          

    def wiener_with_left_censoring_lpdf(self, y, alpha, tau, beta, delta, event):
        self._function('wiener_with_left_censoring_lpdf', [y, alpha, tau, beta, delta, event])          

    def wiener_with_left_censoring_rng(self, y, alpha, tau, beta, delta, event):
        self._function('wiener_with_left_censoring_rng', [y, alpha, tau, beta, delta, event])          

    def wiener_with_right_censoring(self, alpha, tau, beta, delta):
        self._function('wiener_with_right_censoring', [alpha, tau, beta, delta])          

    def wiener_with_right_censoring_lpdf(self, y, alpha, tau, beta, delta, event):
        self._function('wiener_with_right_censoring_lpdf', [y, alpha, tau, beta, delta, event])          

    def wiener_with_right_censoring_rng(self, y, alpha, tau, beta, delta, event):
        self._function('wiener_with_right_censoring_rng', [y, alpha, tau, beta, delta, event])          

    def wishart(self, nu, Sigma):
        self._function('wishart', [nu, Sigma])          

    def wishart_cholesky(self, nu, L_S):
        self._function('wishart_cholesky', [nu, L_S])          

    def wishart_cholesky_lpdf(self, L_W, nu, L_S):
        self._function('wishart_cholesky_lpdf', [L_W, nu, L_S])          

    def wishart_cholesky_lupdf(self, L_W, nu, L_S):
        self._function('wishart_cholesky_lupdf', [L_W, nu, L_S])          

    def wishart_cholesky_rng(self, nu, L_S):
        self._function('wishart_cholesky_rng', [nu, L_S])          

    def wishart_cholesky_with_left_and_right_censoring(self, nu, L_S, event_left, event_right):
        self._function('wishart_cholesky_with_left_and_right_censoring', [nu, L_S, event_left, event_right])          

    def wishart_cholesky_with_left_and_right_censoring_lpdf(self, L_W, nu, L_S, event_left, event_right):
        self._function('wishart_cholesky_with_left_and_right_censoring_lpdf', [L_W, nu, L_S, event_left, event_right])          

    def wishart_cholesky_with_left_and_right_censoring_rng(self, nu, L_S, event_left, event_right):
        self._function('wishart_cholesky_with_left_and_right_censoring_rng', [nu, L_S, event_left, event_right])          

    def wishart_cholesky_with_left_censoring(self, nu, L_S, event):
        self._function('wishart_cholesky_with_left_censoring', [nu, L_S, event])          

    def wishart_cholesky_with_left_censoring_lpdf(self, L_W, nu, L_S, event):
        self._function('wishart_cholesky_with_left_censoring_lpdf', [L_W, nu, L_S, event])          

    def wishart_cholesky_with_left_censoring_rng(self, L_W, nu, L_S, event):
        self._function('wishart_cholesky_with_left_censoring_rng', [L_W, nu, L_S, event])          

    def wishart_cholesky_with_right_censoring(self, nu, L_S):
        self._function('wishart_cholesky_with_right_censoring', [nu, L_S])          

    def wishart_cholesky_with_right_censoring_lpdf(self, L_W, nu, L_S, event):
        self._function('wishart_cholesky_with_right_censoring_lpdf', [L_W, nu, L_S, event])          

    def wishart_cholesky_with_right_censoring_rng(self, L_W, nu, L_S, event):
        self._function('wishart_cholesky_with_right_censoring_rng', [L_W, nu, L_S, event])          

    def wishart_lpdf(self, W, nu, Sigma):
        self._function('wishart_lpdf', [W, nu, Sigma])          

    def wishart_lupdf(self, W, nu, Sigma):
        self._function('wishart_lupdf', [W, nu, Sigma])          

    def wishart_rng(self, nu, Sigma):
        self._function('wishart_rng', [nu, Sigma])          

    def wishart_with_left_and_right_censoring(self, nu, Sigma, event_left, event_right):
        self._function('wishart_with_left_and_right_censoring', [nu, Sigma, event_left, event_right])          

    def wishart_with_left_and_right_censoring_lpdf(self, W, nu, Sigma, event_left, event_right):
        self._function('wishart_with_left_and_right_censoring_lpdf', [W, nu, Sigma, event_left, event_right])          

    def wishart_with_left_and_right_censoring_rng(self, nu, Sigma, event_left, event_right):
        self._function('wishart_with_left_and_right_censoring_rng', [nu, Sigma, event_left, event_right])          

    def wishart_with_left_censoring(self, nu, Sigma, event):
        self._function('wishart_with_left_censoring', [nu, Sigma, event])          

    def wishart_with_left_censoring_lpdf(self, W, nu, Sigma, event):
        self._function('wishart_with_left_censoring_lpdf', [W, nu, Sigma, event])          

    def wishart_with_left_censoring_rng(self, W, nu, Sigma, event):
        self._function('wishart_with_left_censoring_rng', [W, nu, Sigma, event])          

    def wishart_with_right_censoring(self, nu, Sigma):
        self._function('wishart_with_right_censoring', [nu, Sigma])          

    def wishart_with_right_censoring_lpdf(self, W, nu, Sigma, event):
        self._function('wishart_with_right_censoring_lpdf', [W, nu, Sigma, event])          

    def wishart_with_right_censoring_rng(self, W, nu, Sigma, event):
        self._function('wishart_with_right_censoring_rng', [W, nu, Sigma, event])          

    def zeros_array(self, n):
        self._function('zeros_array', [n])          

    def zeros_int_array(self, n):
        self._function('zeros_int_array', [n])          

    def zeros_row_vector(self, n):
        self._function('zeros_row_vector', [n])          
