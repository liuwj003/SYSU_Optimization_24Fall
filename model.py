import numpy as np
import matplotlib.pyplot as plt


# initialize real value of x
real_x = np.zeros(200)
indices = np.random.choice(200,5)
real_x[indices] = np.random.randn(5)  # N(0,1)

# initialize measurement matrix
measure_matrix = [np.random.randn(5,200) for _ in range(10)]

# initialize measurement noise
measure_noise = [np.random.normal(0, 0.1, 5) for _ in range(10)]

# measurement result with noise
measure_result = [np.dot(measure_matrix[i], real_x) + measure_noise[i] for i in range(10)]


def proximal_gradient(x_0, A, b, iters, regu_lambda, step_size):
    """
    Proximal Gradient Method
    :param x_0: variable to be updated
    :param A: measurement matrix
    :param b: measurement result with noise
    :param iters: iterations num
    :param regu_lambda: regularization coefficient
    :param step_size: step size alpha
    :return: updated x, x_list
    """
    iter_list = []
    x = np.copy(x_0)

    for i in range(iters):
        grad = 0
        for j in range(10):
            grad += np.dot(A[j].T, np.dot(A[j], x) - b[j])

        x -= step_size * grad
        # soft threshold
        condlist = [x > step_size * regu_lambda, x < - step_size * regu_lambda]
        choicelist = [x - step_size * regu_lambda, x + step_size * regu_lambda]
        x = np.select(condlist, choicelist, 0)

        iter_list.append(np.copy(x))

    return x, iter_list


def ADMM(x_0, A, b, iters, regu_lambda, step_size):
    """
    Alternating Direction Method of Multipliers
    :param x_0: variable to be updated
    :param A: measurement matrix
    :param b: measurement result with noise
    :param iters: iterations num
    :param regu_lambda: regularization coefficient
    :param step_size: step size c
    :return: updated x, x_list
    """
    iter_list = []
    x = np.copy(x_0)
    y = np.copy(x)
    v = np.zeros(200)

    part1 = step_size * np.eye(200)
    for j in range(10):
        part1 += np.dot(A[j].T, A[j])
    part1 = np.linalg.inv(part1)

    for i in range(iters):
        # update x
        part2 = step_size * y - v
        for j in range(10):
            part2 += np.dot(A[j].T, b[j])

        x = np.dot(part1, part2)

        iter_list.append(np.copy(x))

        # update y
        condlist = [v + step_size * x > regu_lambda, v + step_size * x < - regu_lambda]
        choicelist = [(v - regu_lambda) / step_size + x, (v + regu_lambda) / step_size + x]
        y = np.select(condlist, choicelist, 0)

        # update v
        v += step_size * (x - y)

    return x, iter_list


def subgradient(x_0, A, b, iters, regu_lambda, step_size):
    """
    Subgradient Method
    :param x_0: variable to be updated
    :param A: measurement matrix
    :param b: measurement result with noise
    :param iters: iterations num
    :param regu_lambda: regularization coefficient
    :param step_size: step size c
    :return: updated x, x_list
    """
    iter_list = []

    x = np.copy(x_0)

    for i in range(iters):
        grad_sum = 0
        for j in range(10):
            grad_sum += np.dot(A[j].T, np.dot(A[j], x) - b[j])
        condlist = [x > 0, x == 0, x < 0]
        choicelist = [grad_sum + regu_lambda, grad_sum + np.random.uniform(- regu_lambda, regu_lambda), grad_sum - regu_lambda]
        g_x = np.select(condlist, choicelist)

        if i > 0:
            step_size_k = step_size / (i ** 0.5)
        else:
            step_size_k = step_size

        x -= step_size_k * g_x
        iter_list.append(np.copy(x))

    return x, iter_list


def approximate_sparsity(x, threshold):
    """
    compute the approximate sparsity of x
    :param x:
    :param threshold: if x_i < threshold, we think it is zero
    :return: sparsity
    """
    non_zero_count = np.sum(np.abs(x) > threshold)
    return non_zero_count


def compute_distances(x_list, x_opt, real_x):
    """
    compute ||x_k - x_opt||_2, ||x_k - real_x||_2
    :param x_list:
    :param x_opt:
    :param real_x:
    :return:
    """
    opt_distances = [np.linalg.norm(x - x_opt) for x in x_list]
    real_distances = [np.linalg.norm(x - real_x) for x in x_list]
    return opt_distances, real_distances

def main():
    x_0 = np.zeros(200)
    prox_gd_x_final, prox_gd_x_list = proximal_gradient(x_0, measure_matrix, measure_result, iters=1000, regu_lambda=1, step_size=1e-3)
    admm_x_final, admm_x_list = ADMM(x_0, measure_matrix, measure_result, iters=1000, regu_lambda=1, step_size=1e-1)
    x_opt = admm_x_final  # use ADMM optimal solution as x_opt
    sub_gd_x_final, sub_gd_x_list = subgradient(x_0, measure_matrix, measure_result, iters=1000, regu_lambda=1, step_size=1e-2)

    # compute distances
    prox_gd_opt_distances, prox_gd_real_distances = compute_distances(prox_gd_x_list, x_opt, real_x)
    admm_opt_distances, admm_real_distances = compute_distances(admm_x_list, x_opt, real_x)
    sub_gd_opt_distances, sub_gd_real_distances = compute_distances(sub_gd_x_list, x_opt, real_x)

    print(real_x)
    print(prox_gd_x_final)
    print(admm_x_final)
    print(sub_gd_x_final)

    plt.figure(figsize=(12, 8))

    # proximal gradient method
    plt.plot(prox_gd_opt_distances, label="ProxGD: $\\|x_k - x_{opt}\\|_2$")
    plt.plot(prox_gd_real_distances, label="ProxGD: $\\|x_k - x_{real}\\|_2$")

    # ADMM
    plt.plot(admm_opt_distances, label="ADMM: $\\|x_k - x_{opt}\\|_2$")
    plt.plot(admm_real_distances, label="ADMM: $\\|x_k - x_{real}\\|_2$")

    # Subgradient
    plt.plot(sub_gd_opt_distances, label="SubGD: $\\|x_k - x_{opt}\\|_2$")
    plt.plot(sub_gd_real_distances, label="SubGD: $\\|x_k - x_{real}\\|_2$")

    plt.xlabel("Iteration")
    plt.ylabel("Distance (l2-norm)")
    plt.title("Distance to $x_{opt}$ and $x_{real}$ over iterations")
    plt.legend()
    plt.grid()
    plt.savefig(f"Distances.png")
    plt.show()

    regu_lambdas = [1e-2, 1e-1, 1, 10, 100]
    # regu_lambdas = [1e-2, 1e-1, 1, 10]

    prox_gd_sparsities = []
    admm_sparsities = []
    sub_gd_sparsities = []

    for regu_lambda in regu_lambdas:
        # Proximal Gradient Method
        prox_gd_x_final, _ = proximal_gradient(x_0, measure_matrix, measure_result, iters=2000, regu_lambda=regu_lambda, step_size=1e-3)
        prox_gd_sparsities.append(approximate_sparsity(prox_gd_x_final, threshold=1e-2))

        # ADMM
        admm_x_final, _ = ADMM(x_0, measure_matrix, measure_result, iters=2000, regu_lambda=regu_lambda, step_size=1e-1)
        admm_sparsities.append(approximate_sparsity(admm_x_final, threshold=1e-2))

        # Subgradient Method
        sub_gd_x_final, _ = subgradient(x_0, measure_matrix, measure_result, iters=2000, regu_lambda=regu_lambda, step_size=1e-2)
        sub_gd_sparsities.append(approximate_sparsity(sub_gd_x_final, threshold=1e-2))

    plt.figure(figsize=(12, 8))
    plt.plot(regu_lambdas, prox_gd_sparsities, label="Proximal Gradient", marker='o')
    plt.plot(regu_lambdas, admm_sparsities, label="ADMM", marker='^')
    plt.plot(regu_lambdas, sub_gd_sparsities, label="Subgradient", marker='s')

    plt.xscale('log')
    plt.xlabel("Regularization Coefficient $\\lambda$")
    plt.ylabel("Approximate Sparsity")
    plt.title("Sparsity-vs-Regularization $\\lambda$")
    plt.legend()
    plt.grid()
    plt.savefig(f"Sparsity_vs_Lambda.png")
    plt.show()


if __name__ == "__main__":
    main()