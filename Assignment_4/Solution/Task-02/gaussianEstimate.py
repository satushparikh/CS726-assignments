import numpy as np
import matplotlib.pyplot as plt

def branin_hoo(x):
    """Calculate the Branin-Hoo function value for given input."""
    x = np.atleast_2d(x)
    x1, x2 = x[:, 0], x[:, 1]

    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)

    return a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s

# Kernel Functions (Students implement)
def rbf_kernel(x1, x2, length_scale=1.0, sigma_f=1.0):
    """Compute the RBF kernel."""
    # Compute the squared Euclidean distance between x1 and x2
    sqdist = np.sum(x1**2, 1).reshape(-1,1) + np.sum(x2**2, 1) - 2*np.dot(x1, x2.T)
    # Compute the RBF kernel
    k = sigma_f**2 * np.exp(-0.5 * sqdist / length_scale**2)
    return k

def matern_kernel(x1, x2, length_scale=1.0, sigma_f=1.0, nu=1.5):
    """Compute the Matérn kernel (nu=1.5)."""
    """Compute the Matérn kernel (ν=1.5) between two sets of input points."""
    # Ensure input arrays are 2D
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)

    # Compute pairwise Euclidean distances
    sqdist = np.sum(x1**2, axis=1).reshape(-1, 1) +  np.sum(x2**2, axis=1) - 2 * np.dot(x1, x2.T)
    dists = np.sqrt(np.maximum(sqdist, 1e-12))  # Numerical stability

    sqrt3 = np.sqrt(3)
    scaled_dist = sqrt3 * dists / length_scale

    k = sigma_f**2 * (1 + scaled_dist) * np.exp(-scaled_dist)
    return k

def rational_quadratic_kernel(x1, x2, length_scale=1.0, sigma_f=1.0, alpha=1.0):
    """Compute the Rational Quadratic kernel."""
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    
    # Squared Euclidean distance
    sqdist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    
    # Rational Quadratic kernel
    k = sigma_f**2 * (1 + sqdist / (2 * alpha * length_scale**2)) ** (-alpha)
    
    return k

def log_marginal_likelihood(x_train, y_train, kernel_func, length_scale, sigma_f, noise=1e-4):
    """Compute the log-marginal likelihood."""
    pass

def optimize_hyperparameters(x_train, y_train, kernel_func, noise=1e-4):
    """Optimize hyperparameters using grid search."""    
    pass

def gaussian_process_predict(x_train, y_train, x_test, kernel_func, length_scale=1.0, sigma_f=1.0, noise=1e-4):
    """Perform GP prediction."""

    pass

# Acquisition Functions (Simplified, no erf)
def logistic_cdf(z):
    """Compute the logistic CDF."""
    # Approximate Phi(z) = 1 / (1 + exp(-1.702 * z))
    return 1 / (1 + np.exp(-1.702 * z))

def logistic_pdf(z):
    """Derivative of the logistic CDF."""
    exp_term = np.exp(-1.702 * z)
    return 1.702 * exp_term / ((1.0 + exp_term)**2)

def expected_improvement(mu, sigma, y_best, xi=0.01):
    """Compute Expected Improvement acquisition function."""
    # Approximate Phi(z) = 1 / (1 + exp(-1.702 * z))
    """ 
    EI(x) = (mu(x) - y_best - xi) * Phi(z) + sigma(x) * phi(z)
    where z = (mu(x) - y_best - xi) / sigma(x), Phi is the cumulative distribution function and phi is the probability density function
    """
    # ensure numerical stability
    sigma = np.maximum(sigma, 1e-8)
    # compute z
    z = (mu - y_best - xi) / sigma
    ei = (mu - y_best - xi) * logistic_cdf(z) + sigma * logistic_pdf(z)
    ei[sigma < 1e-8] = 0
    return ei

def probability_of_improvement(mu, sigma, y_best, xi=0.01):
    """Compute Probability of Improvement acquisition function."""
    # Approximate Phi(z) = 1 / (1 + exp(-1.702 * z))
    with np.errstate(divide='ignore',invalid='ignore'):
        # numpy context manager to ignore divide by zero and invalid values
        z = (mu - y_best - xi) / sigma 
        phi_approx = logistic_cdf(z)
        # where sigma is close to 0, manually set probability of improvement to 0 or 1 pl
        phi_approx = np.where(sigma < 1e-8, (mu > y_best +xi).astype(float), phi_approx)
    pass

def plot_graph(x1_grid, x2_grid, z_values, x_train, title, filename):
    """Create and save a contour plot."""
    

def main():
    """Main function to run GP with kernels, sample sizes, and acquisition functions."""
    np.random.seed(0)
    n_samples_list = [10, 20, 50, 100]
    kernels = {
        'rbf': (rbf_kernel, 'RBF'),
        'matern': (matern_kernel, 'Matern (nu=1.5)'),
        'rational_quadratic': (rational_quadratic_kernel, 'Rational Quadratic')
    }
    acquisition_strategies = {
        'EI': expected_improvement,
        'PI': probability_of_improvement
    }
    
    x1_test = np.linspace(-5, 10, 100)
    x2_test = np.linspace(0, 15, 100)
    x1_grid, x2_grid = np.meshgrid(x1_test, x2_test)
    x_test = np.c_[x1_grid.ravel(), x2_grid.ravel()]
    true_values = np.array([branin_hoo([x1, x2]) for x1, x2 in x_test]).reshape(x1_grid.shape)
    
    for kernel_name, (kernel_func, kernel_label) in kernels.items():
        for n_samples in n_samples_list:
            x_train = np.random.uniform(low=[-5, 0], high=[10, 15], size=(n_samples, 2))
            y_train = np.array([branin_hoo(x) for x in x_train])
            
            print(f"\nKernel: {kernel_label}, n_samples = {n_samples}")
            length_scale, sigma_f, noise = optimize_hyperparameters(x_train, y_train, kernel_func)
            
            for acq_name, acq_func in acquisition_strategies.items():
                x_train_current = x_train.copy()
                y_train_current = y_train.copy()
                
                y_mean, y_std = gaussian_process_predict(x_train_current, y_train_current, x_test, 
                                                        kernel_func, length_scale, sigma_f, noise)
                y_mean_grid = y_mean.reshape(x1_grid.shape)
                y_std_grid = y_std.reshape(x1_grid.shape)
                
                if acq_func is not None:
                    # Hint: Find y_best, apply acq_func, select new point, update training set, recompute GP
                    pass
                
                acq_label = '' if acq_name == 'None' else f', Acq={acq_name}'
                plot_graph(x1_grid, x2_grid, true_values, x_train_current,
                          f'True Branin-Hoo Function (n={n_samples}, Kernel={kernel_label}{acq_label})',
                          f'true_function_{kernel_name}_n{n_samples}_{acq_name}.png')
                plot_graph(x1_grid, x2_grid, y_mean_grid, x_train_current,
                          f'GP Predicted Mean (n={n_samples}, Kernel={kernel_label}{acq_label})',
                          f'gp_mean_{kernel_name}_n{n_samples}_{acq_name}.png')
                plot_graph(x1_grid, x2_grid, y_std_grid, x_train_current,
                          f'GP Predicted Std Dev (n={n_samples}, Kernel={kernel_label}{acq_label})',
                          f'gp_std_{kernel_name}_n{n_samples}_{acq_name}.png')

if __name__ == "__main__":
    main()