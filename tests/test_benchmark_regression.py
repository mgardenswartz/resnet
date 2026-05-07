import jax
import jax.numpy as jnp
from jax_resnet.resnet import resnet_network, init_resnet_weights

def mse_loss(theta, x, y, h_act_func, o_act_func):
    # Vectorize the network to handle batch inputs
    batched_net = jax.vmap(lambda xi: resnet_network(
        theta=theta, x=xi, d_in=1, hidden_width=16, d_out=1,
        b=2, k_0=2, k_i=2, h_act_func=h_act_func, o_act_func=o_act_func, shortcut_act_func="linear"
    ))
    preds = batched_net(x)
    return jnp.mean((preds - y)**2)

# CRITICAL: Tell JAX not to trace the string arguments
@jax.jit(static_argnames=['h_act_func', 'o_act_func'])
def update(theta, x, y, h_act_func, o_act_func, lr):
    loss, grads = jax.value_and_grad(mse_loss)(theta, x, y, h_act_func, o_act_func)
    return theta - lr * grads, loss

def run_experiment(h_init, o_init, h_act_func, o_act_func, lr):
    key = jax.random.PRNGKey(42)
    
    # 1D regression task: y = sin(x)
    x = jnp.linspace(-3, 3, 100).reshape(-1, 1)
    y = jnp.sin(x)

    theta = init_resnet_weights(
        key=key, d_in=1, hidden_width=16, d_out=1,
        b=2, k_0=2, k_i=2, h_method=h_init, o_method=o_init
    )

    for _ in range(1000):
        theta, loss = update(theta, x, y, h_act_func, o_act_func, lr)

    act_name = f"h={h_act_func}, o={o_act_func}"
    print(f"H-Init: {h_init.ljust(6)} | O-Init: {o_init.ljust(6)} | LR: {lr:.4f} | Act: {act_name.ljust(30)} | Final MSE: {loss:.4f}")

if __name__ == "__main__":
    print("Running Advanced Regression Benchmarks...\n")
    
    # 1. The Classic Control Standard
    run_experiment('xavier', 'xavier', 'tanh', 'linear', 0.01)
    
    # 2. The Slow-but-Safe Unbounded
    run_experiment('he', 'he', 'leaky_relu_approx', 'linear', 0.0005)
    
    # 3. The Lab's Hybrid Setup (Swish inner + He, Tanh outer + Xavier)
    run_experiment('he', 'xavier', 'swish', 'tanh', 0.01)
    
    # 4. Pure Swish (to compare against the hybrid)
    try:
        run_experiment('he', 'he', 'swish', 'linear', 0.0005)
    except Exception as e:
        print(f"H-Init: he     | O-Init: he     | LR: 0.0005 | Act: h=swish, o=linear               | Crashed (Exploded)")