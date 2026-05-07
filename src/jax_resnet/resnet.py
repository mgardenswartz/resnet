import jax
import jax.numpy as jnp


ACTIVATION_MAP = {
    "linear": 0,
    "swish": 1,
    "tanh": 2,
    "logistic": 3,
    "softplus": 4,
    "leaky_relu_approx": 5
}

# Constants for Class A_quad activation functions
SOFTPLUS_R = 1.0
LEAKY_RELU_K = 10.0
LEAKY_RELU_R = 0.1


def _activate(
    s: jax.Array,
    act_idx: jax.Array,
) -> jax.Array:
    return jax.lax.switch(
        act_idx,
        [
            lambda x: x,                                                # 0: linear
            lambda x: x * jax.nn.sigmoid(x),                            # 1: swish
            lambda x: jnp.tanh(x),                                      # 2: tanh
            lambda x: jax.nn.sigmoid(x),                                # 3: logistic
            lambda x: (1.0 / SOFTPLUS_R) * jax.nn.softplus(SOFTPLUS_R * x), 
            lambda x: (LEAKY_RELU_R * x + jax.nn.softplus((1.0 - LEAKY_RELU_R) * LEAKY_RELU_K * x)) / LEAKY_RELU_K 
        ],
        s
    )


def get_total_parameters(
    d_in: int,
    hidden_width: int,
    d_out: int,
    b: int,
    k_0: int,
    k_i: int,
) -> int:
    total = _get_block_parameters(d_in, hidden_width, d_out, k_0)
    total += b * _get_block_parameters(d_out, hidden_width, d_out, k_i)
    return total


def _get_block_parameters(
    d_in: int,
    hidden_width: int,
    d_out: int,
    k: int,
) -> int:
    if k == 0:
        return (d_in + 1) * d_out
    p_in = (d_in + 1) * hidden_width
    p_hidden = (k - 1) * (hidden_width + 1) * hidden_width if k > 0 else 0
    p_out = (hidden_width + 1) * d_out
    return p_in + p_hidden + p_out


def _mlp_block(
    v: jax.Array, 
    theta: jax.Array, 
    in_dim: int, 
    hidden_width: int, 
    out_dim: int, 
    k: int, 
    h_act_idx: jax.Array,
    o_act_idx: jax.Array,
) -> jax.Array:
    idx = 0
    
    if k == 0:
        v0_size = (in_dim + 1) * out_dim
        v0 = jnp.reshape(theta[idx:idx + v0_size], (in_dim + 1, out_dim), order='F')
        return jnp.dot(v, v0)

    v0_size = (in_dim + 1) * hidden_width
    v0 = jnp.reshape(theta[idx:idx + v0_size], (in_dim + 1, hidden_width), order='F')
    idx += v0_size
    phi_j = jnp.dot(v, v0)
    
    for _ in range(k - 1):
        v_size = (hidden_width + 1) * hidden_width
        v_j = jnp.reshape(theta[idx:idx + v_size], (hidden_width + 1, hidden_width), order='F')
        idx += v_size
        sigma_a = jnp.append(_activate(phi_j, h_act_idx), 1.0) 
        phi_j = jnp.dot(sigma_a, v_j)
        
    vk_size = (hidden_width + 1) * out_dim
    vk = jnp.reshape(theta[idx:idx + vk_size], (hidden_width + 1, out_dim), order='F')
    sigma_a_out = jnp.append(_activate(phi_j, o_act_idx), 1.0) 
    
    return jnp.dot(sigma_a_out, vk)


def resnet_network(
    theta: jax.Array,
    x: jax.Array,
    d_in: int,
    hidden_width: int,
    d_out: int,
    b: int,          
    k_0: int,        
    k_i: int,        
    h_act_idx: jax.Array,
    o_act_idx: jax.Array,
    shortcut_act_idx: jax.Array
) -> jax.Array:
    idx = 0
    
    b0_params = _get_block_parameters(d_in, hidden_width, d_out, k_0)
    theta_0 = theta[idx:idx + b0_params]
    idx += b0_params
    
    x_a = jnp.append(x, 1.0)
    kappa = _mlp_block(x_a, theta_0, d_in, hidden_width, d_out, k_0, h_act_idx, o_act_idx)
    
    bi_params = _get_block_parameters(d_out, hidden_width, d_out, k_i)
    
    for _ in range(b):
        theta_i = theta[idx:idx + bi_params]
        idx += bi_params
        
        psi_out = jnp.append(_activate(kappa, shortcut_act_idx), 1.0)
        kappa = kappa + _mlp_block(psi_out, theta_i, d_out, hidden_width, d_out, k_i, h_act_idx, o_act_idx)
        
    return kappa


def compute_jacobian(
    theta: jax.Array,
    x: jax.Array,
    d_in: int,
    hidden_width: int,
    d_out: int,
    b: int,
    k_0: int,
    k_i: int,
    h_act_idx: jax.Array,
    o_act_idx: jax.Array,
    shortcut_act_idx: jax.Array
) -> jax.Array:
    return jax.jacrev(resnet_network, argnums=0)(
        theta, x, d_in, hidden_width, d_out, b, k_0, k_i, h_act_idx, o_act_idx, shortcut_act_idx
    )


def _init_layer(
    key: jax.Array,
    fan_in: int,
    fan_out: int,
    method: str
) -> jax.Array:
    match method:
        case 'he':
            std = jnp.sqrt(2.0 / fan_in)
            w = std * jax.random.normal(key, (fan_in, fan_out))
        case 'xavier':
            std = jnp.sqrt(2.0 / (fan_in + fan_out))
            w = std * jax.random.normal(key, (fan_in, fan_out))
        case 'orthogonal':
            w = jax.nn.initializers.orthogonal()(key, (fan_in, fan_out))
        case 'zero':
            w = jnp.zeros((fan_in, fan_out))
        case _:
            raise ValueError(f"Unknown method: {method}")

    b = jnp.zeros((1, fan_out))
    return jnp.vstack([w, b]).flatten(order='F')


def init_resnet_weights(
    key: jax.Array,
    d_in: int,
    hidden_width: int,
    d_out: int,
    b: int,
    k_0: int,
    k_i: int,
    h_method: str,
    o_method: str
) -> jax.Array:
    theta_list = []
    
    def _build_block(
        key: jax.Array,
        block_in: int,
        block_out: int,
        k: int,
        is_residual: bool
    ) -> list:
        keys = jax.random.split(key, max(1, k))
        block_theta = []
        
        if k == 0:
            layer_method = 'zero' if is_residual else o_method
            block_theta.append(_init_layer(keys[0], block_in, block_out, layer_method))
            return block_theta
            
        block_theta.append(_init_layer(keys[0], block_in, hidden_width, h_method))
        
        for j in range(1, k):
            block_theta.append(_init_layer(keys[j], hidden_width, hidden_width, h_method))
            
        layer_method = 'zero' if is_residual else o_method
        block_theta.append(_init_layer(keys[-1], hidden_width, block_out, layer_method))
        
        return block_theta

    keys = jax.random.split(key, b + 1)
    
    theta_list.extend(_build_block(keys[0], d_in, d_out, k_0, False))
    
    for i in range(b):
        theta_list.extend(_build_block(keys[i+1], d_out, d_out, k_i, True))
        
    return jnp.concatenate(theta_list)
