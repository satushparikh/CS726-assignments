import numpy as np
import matplotlib.pyplot as plt
import os

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
    # Compute the kernel matrix
    K = kernel_func(x_train, x_train, length_scale, sigma_f)
    
    # Add noise to the diagonal
    K += noise * np.eye(len(x_train))
    
    # Compute the log determinant term K=LL^T
    L = np.linalg.cholesky(K)
    log_det = 2 * np.sum(np.log(np.diag(L))) # log|K|=log|L|+log|L^T|=2log|L|. L is lower triangular matrix
    
    # Compute the quadratic term
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    quadratic_term = y_train.T @ alpha
    
    # p(y|X,theta)=N(y|0,K(X,X)+sigma^2I)
    # log p(y|X,theta)=-0.5*(y^T(K(X,X)+sigma^2I)^-1y+log|K(X,X)+sigma^2I|+n*log(2*pi))
    # Compute the log marginal likelihood
    log_likelihood = -0.5 * (quadratic_term + log_det + len(x_train) * np.log(2 * np.pi))
    
    return log_likelihood

def optimize_hyperparameters(x_train, y_train, kernel_func, noise=1e-4):
    """Optimize hyperparameters using grid search."""    
    # Define the grid of hyperparameters to search
    length_scales = np.logspace(-2, 2, 20)
    sigma_fs = np.logspace(-2, 2, 20)
    
    best_log_likelihood = -np.inf
    best_length_scale = 1.0
    best_sigma_f = 1.0
    
    # Grid search over hyperparameters
    for length_scale in length_scales:
        for sigma_f in sigma_fs:
            # Compute log marginal likelihood for current hyperparameters
            log_likelihood = log_marginal_likelihood(x_train, y_train, kernel_func, 
                                                   length_scale, sigma_f, noise)
            
            # Update best hyperparameters if current ones are better
            if log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_length_scale = length_scale
                best_sigma_f = sigma_f
    
    return best_length_scale, best_sigma_f, noise

def gaussian_process_predict(x_train, y_train, x_test, kernel_func, length_scale=1.0, sigma_f=1.0, noise=1e-4):
    """Perform GP prediction."""
    """
    Perform Gaussian Process prediction for the test points x_test given the training points x_train and their corresponding values y_train.
    """
    # Compute kernel matrices
    # Compute the Kernel matrix of training inputs n*n
    K = kernel_func(x_train, x_train, length_scale, sigma_f) + noise * np.eye(len(x_train))
    # Compute the Kernel matrix of training inputs and test inputs n*m
    K_star = kernel_func(x_train, x_test, length_scale, sigma_f) 
    # Compute the Kernel matrix of test inputs m*m
    K_star_star = kernel_func(x_test, x_test, length_scale, sigma_f) + noise * np.eye(len(x_test))
    
    try:
        # Compute Cholesky decomposition with error handling
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        # If Cholesky fails, add more noise to diagonal
        K += 1e-6 * np.eye(len(x_train))
        L = np.linalg.cholesky(K)
    
    # Solving for alpha = K^-1 @ y using Cholesky decomposition
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    
    # Compute predictive mean with error handling
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        mu = K_star.T @ alpha
        # Clip extreme values
        mu = np.clip(mu, -1e10, 1e10)
    
    # Compute predictive variance
    # predictive variance = K_star_star - K_star^T @ K^-1 @ K_star
    v = np.linalg.solve(L, K_star)
    var = K_star_star - v.T @ v
    
    # Ensure numerical stability
    var = np.maximum(var, 0)
    
    return mu, np.sqrt(np.diag(var))

# Acquisition Functions (Simplified, no erf)
def logistic_cdf(z):
    """Compute the logistic CDF."""
    # Approximate Phi(z) = 1 / (1 + exp(-1.702 * z))
    z = np.asarray(z)
    # clip z to avoid overflow
    z = np.clip(z, -10 / 1.702, 10 / 1.702)
    return 1 / (1 + np.exp(-1.702 * z))


def logistic_pdf(z):
    """Stable approximation of logistic PDF."""
    a = 1.702
    z = np.asarray(z)

    # Check if z is too large or too small (i.e., would cause overflow)
    # if np.any(z > 10 / a) or np.any(z < -10 / a):
        # raise AssertionError(f"Overflow risk: z values are out of safe bounds (z={z})")

    # Clip z to avoid overflow in further calculations
    z = np.clip(z, -10 / a, 10 / a)  # Or a tighter bound like [-300/a, 300/a]

    exp_term = np.where(
        z >= 0,
        np.exp(-a * z),
        1.0 / np.exp(a * z)
    )

    return a * exp_term / ((1.0 + exp_term)**2)

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
    z = (mu - y_best - xi) / sigma[:, np.newaxis]
    ei = (mu - y_best - xi) * logistic_cdf(z) + sigma[:, np.newaxis] * logistic_pdf(z)
    ei[sigma < 1e-8] = 0
    return ei

def probability_of_improvement(mu, sigma, y_best, xi=0.01):
    """Compute Probability of Improvement acquisition function."""
    # Approximate Phi(z) = 1 / (1 + exp(-1.702 * z))
    with np.errstate(divide='ignore',invalid='ignore'):
        # numpy context manager to ignore divide by zero and invalid values
        z = (mu - y_best - xi) / sigma[:, np.newaxis]
        phi_approx = logistic_cdf(z)
        # where sigma is close to 0, manually set probability of improvement to 0 or 1 pl
        phi_approx = np.where(sigma[:, np.newaxis] < 1e-8, (mu > y_best +xi).astype(float), phi_approx)
    return phi_approx

def plot_graph(x1_grid, x2_grid, true_values, y_mean_grid, y_std_grid, x_train, title, filename):
    """Create and save a figure with three subplots."""
    plt.figure(figsize=(15, 5))
    
    # First subplot: True Branin-Hoo function
    plt.subplot(131)
    contour1 = plt.contourf(x1_grid, x2_grid, true_values, levels=20, cmap='viridis')
    plt.colorbar(contour1, label='Function Value')
    plt.scatter(x_train[:, 0], x_train[:, 1], c='red', marker='x', s=100, label='Training Points')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('True Branin-Hoo Function')
    plt.legend()
    
    # Second subplot: GP Predicted Mean
    plt.subplot(132)
    contour2 = plt.contourf(x1_grid, x2_grid, y_mean_grid, levels=20, cmap='viridis')
    plt.colorbar(contour2, label='Function Value')
    plt.scatter(x_train[:, 0], x_train[:, 1], c='red', marker='x', s=100, label='Training Points')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('GP Predicted Mean')
    plt.legend()
    
    # Third subplot: GP Predicted Std Dev
    plt.subplot(133)
    contour3 = plt.contourf(x1_grid, x2_grid, y_std_grid, levels=20, cmap='viridis')
    plt.colorbar(contour3, label='Standard Deviation')
    plt.scatter(x_train[:, 0], x_train[:, 1], c='red', marker='x', s=100, label='Training Points')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('GP Predicted Uncertainty')
    plt.legend()
    
    # Add main title
    plt.suptitle(title, y=1.05)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

def plot_progression(x1_grid, x2_grid, true_values, all_results, kernel_label, acq_name, filename):
    """Create a figure showing progression of GP predictions for different sample sizes."""
    n_samples_list = list(all_results.keys())
    n_rows = len(n_samples_list)
    
    plt.figure(figsize=(15, 5*n_rows))
    
    for i, n_samples in enumerate(n_samples_list):
        y_mean_grid, y_std_grid, x_train = all_results[n_samples]
        
        # First subplot: True Branin-Hoo function (only for first row)
        if i == 0:
            plt.subplot(n_rows, 3, 1)
            contour1 = plt.contourf(x1_grid, x2_grid, true_values, levels=20, cmap='viridis')
            plt.colorbar(contour1, label='Function Value')
            plt.scatter(x_train[:, 0], x_train[:, 1], c='red', marker='x', s=100, label='Training Points')
            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.title('True Branin-Hoo Function')
            plt.legend()
        
        # Second subplot: GP Predicted Mean
        plt.subplot(n_rows, 3, 3*i + 2)
        contour2 = plt.contourf(x1_grid, x2_grid, y_mean_grid, levels=20, cmap='viridis')
        plt.colorbar(contour2, label='Function Value')
        plt.scatter(x_train[:, 0], x_train[:, 1], c='red', marker='x', s=100, label='Training Points')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title(f'GP Predicted Mean (n={n_samples})')
        plt.legend()
        
        # Third subplot: GP Predicted Std Dev
        plt.subplot(n_rows, 3, 3*i + 3)
        contour3 = plt.contourf(x1_grid, x2_grid, y_std_grid, levels=20, cmap='viridis')
        plt.colorbar(contour3, label='Standard Deviation')
        plt.scatter(x_train[:, 0], x_train[:, 1], c='red', marker='x', s=100, label='Training Points')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title(f'GP Predicted Uncertainty (n={n_samples})')
        plt.legend()
    
    # Add main title
    plt.suptitle(f'GP Results Progression (Kernel={kernel_label}, Acq={acq_name})', y=1.05)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    """Main function to run GP with kernels, sample sizes, and acquisition functions."""
    # Create a directory for plots if it doesn't exist
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(current_dir, 'gp_plots')
    
    # Check if directory exists, if not create it
    if not os.path.exists(plot_dir):
        print(f"Plot directory does not exist. Creating directory at: {plot_dir}")
        try:
            os.makedirs(plot_dir)
            print("Directory created successfully!")
        except Exception as e:
            print(f"Error creating directory: {e}")
            return
    else:
        print(f"Plot directory already exists at: {plot_dir}")
    
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
        for acq_name, acq_func in acquisition_strategies.items():
            # Store results for all sample sizes
            all_results = {}
            
            for n_samples in n_samples_list:
                x_train = np.random.uniform(low=[-5, 0], high=[10, 15], size=(n_samples, 2))
                y_train = np.array([branin_hoo(x) for x in x_train])
                
                print(f"\nKernel: {kernel_label}, n_samples = {n_samples}")
                length_scale, sigma_f, noise = optimize_hyperparameters(x_train, y_train, kernel_func)
                
                x_train_current = x_train.copy()
                y_train_current = y_train.copy()
                
                y_mean, y_std = gaussian_process_predict(x_train_current, y_train_current, x_test, 
                                                        kernel_func, length_scale, sigma_f, noise)
                y_mean_grid = y_mean.reshape(x1_grid.shape)
                y_std_grid = y_std.reshape(x1_grid.shape)
                
                if acq_func is not None:
                    y_best = np.max(y_train_current)
                    acq_values = acq_func(y_mean, y_std, y_best)
                    x_new = x_test[np.argmax(acq_values)]
                    x_train_current = np.vstack([x_train_current, x_new])
                    y_train_current = np.vstack([y_train_current, branin_hoo(x_new)])
                    y_mean, y_std = gaussian_process_predict(x_train_current, y_train_current, x_test, 
                                                        kernel_func, length_scale, sigma_f, noise)
                
                # Store results for this sample size
                all_results[n_samples] = (y_mean_grid, y_std_grid, x_train_current)
            
            # Create progression plot for this kernel and acquisition function
            progression_path = os.path.join(plot_dir, f'progression_{kernel_name}_{acq_name}.png')
            print(f"\nSaving progression plot for {kernel_label} with {acq_name}:")
            print(f"Progression plot: {progression_path}")
            
            try:
                plot_progression(x1_grid, x2_grid, true_values, all_results, kernel_label, acq_name, progression_path)
            except Exception as e:
                print(f"Error saving progression plot: {e}")

if __name__ == "__main__":
    main()