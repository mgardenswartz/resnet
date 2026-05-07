import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
from jax_resnet import resnet_network, compute_jacobian

def test_resnet_verification():
    d_in = 3
    d_out = 3
    hidden_width = 2
    b = 2
    k_0 = 1
    k_i = 1
    
    # Hand calculations strictly apply tanh everywhere.
    h_act = 'tanh' 
    o_act = 'tanh' 
    shortcut_act = 'tanh'
    
    x = jnp.array([0.1, -0.5, 0.25])
    
    theta_0 = jnp.array([-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    theta_1 = jnp.array([0.8, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6])
    theta_2 = jnp.array([0.5, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, -0.7, 0.2, 0.1, 0.0, -0.1])
    theta = jnp.concatenate([theta_0, theta_1, theta_2])

    expected_fwd = np.array([1.262638, -0.071262, -0.90333])
    
    expected_jac_t0 = np.array([
        [0.013932, -0.069658, 0.034829, 0.13932, 0.042308, -0.21154, 0.10577, 0.42308, -0.67519, -0.19344, 1.555, -0.26985, -0.077313, 0.62148, -0.17605, -0.050439, 0.40546],
        [0.0027625, -0.013813, 0.0069063, 0.027625, 0.0045395, -0.022698, 0.011349, 0.045395, 0.11529, 0.033032, -0.26553, -0.2689, -0.07704, 0.61929, 0.10108, 0.02896, -0.2328],
        [0.032987, -0.16493, 0.082467, 0.32987, 0.044057, -0.22029, 0.11014, 0.44057, 0.10081, 0.028884, -0.23219, 0.086875, 0.02489, -0.20008, -0.36727, -0.10522, 0.84586]
    ])
    
    expected_jac_t1 = np.array([
        [0.0026956, 0.0050941, 0.0073259, 0.01893, -0.017564, -0.033191, -0.047733, -0.12334, 1.4154, 0.9484, 1.6127, 0.5818, 0.38984, 0.66287, 0.38138, 0.25554, 0.43452],
        [-0.00081657, -0.0015431, -0.0022192, -0.0057343, -0.0039445, -0.007454, -0.01072, -0.0277, -0.21493, -0.14402, -0.24488, 0.55894, 0.37452, 0.63682, -0.19271, -0.12912, -0.21956],
        [-0.013246, -0.025032, -0.036, -0.093022, -0.04794, -0.090595, -0.13029, -0.33666, 0.033897, 0.022713, 0.03862, 0.02959, 0.019827, 0.033713, 0.89843, 0.60199, 1.0236]
    ])
    
    expected_jac_t2 = np.array([
        [0.2243, -0.1402, -0.4252, 0.6216, 0.2028, -0.1268, -0.3846, 0.5622, 0.3347, 0.2509, 1, 0, 0, 0, 0, 0, 0],
        [0.1282, -0.0801, -0.2430, 0.3552, -0.2366, 0.1480, 0.4487, -0.6559, 0, 0, 0, 0.3347, 0.2509, 1, 0, 0, 0],
        [0.0320, -0.0200, -0.0607, 0.0888, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3347, 0.2509, 1]
    ])

    fwd_out = resnet_network(theta, x, d_in, hidden_width, d_out, b, k_0, k_i, h_act, o_act, shortcut_act)
    assert_allclose(fwd_out, expected_fwd, rtol=1e-4, atol=1e-4)

    jac_out = compute_jacobian(theta, x, d_in, hidden_width, d_out, b, k_0, k_i, h_act, o_act, shortcut_act)
    
    p0 = len(theta_0)
    p1 = len(theta_1)
    p2 = len(theta_2)
    
    jac_t0 = jac_out[:, 0:p0]
    jac_t1 = jac_out[:, p0:p0+p1]
    jac_t2 = jac_out[:, p0+p1:p0+p1+p2]

    assert_allclose(jac_t0, expected_jac_t0, rtol=1e-3, atol=1e-3)
    assert_allclose(jac_t1, expected_jac_t1, rtol=1e-3, atol=1e-3)
    assert_allclose(jac_t2, expected_jac_t2, rtol=1e-3, atol=1e-3)
    
    print("Verification successful. Code matches hand calculations.")

if __name__ == "__main__":
    test_resnet_verification()
