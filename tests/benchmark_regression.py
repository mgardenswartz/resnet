import jax
import jax.numpy as jnp
from src.jax_resnet.resnet import resnet_network, init_resnet_weights

def mse_loss(theta, x, y, h_act_idx, o_act_idx):
    # Vectorize the network to handle batch inputs
    batched_net = jax.vmap(lambda xi: resnet_network(
        theta=theta, x=xi, d_in=1, hidden_width=16, d_out=1,
        b=2, k_0=2, k_i=2, h_act_idx=h_act_idx, o_act_idx=o_act_idx, shortcut_act_idx=0
    ))
    preds = batched_net(x)
    return jnp.mean((preds - y)**2)

@jax.jit
def update(theta, x, y, h_act_idx, o_act_idx, lr):
    loss, grads = jax.value_and_grad(mse_loss)(theta, x, y, h_act_idx, o_act_idx)
    return theta - lr * grads, loss

def run_experiment(h_init, o_init, hidden_act, outer_act, act_name, lr):
    key = jax.random.PRNGKey(42)
    
    # 1D regression task: y = sin(x)
    x = jnp.linspace(-3, 3, 100).reshape(-1, 1)
    y = jnp.sin(x)

    theta = init_resnet_weights(
        key=key, d_in=1, hidden_width=16, d_out=1,
        b=2, k_0=2, k_i=2, h_method=h_init, o_method=o_init
    )

    for _ in range(1000):
        theta, loss = update(theta, x, y, hidden_act, outer_act, lr)

    print(f"H-Init: {h_init.ljust(6)} | O-Init: {o_init.ljust(6)} | LR: {lr:.4f} | Act: {act_name.ljust(25)} | Final MSE: {loss:.4f}")

if __name__ == "__main__":
    print("Running Advanced Regression Benchmarks...\n")
    
    # 1. The Classic Control Standard
    run_experiment('xavier', 'xavier', 2, 0, 'h=tanh, o=linear', 0.01)
    
    # 2. The Slow-but-Safe Unbounded
    run_experiment('he', 'he', 5, 0, 'h=leaky, o=linear', 0.0005)
    
    # 3. The Lab's Hybrid Setup (Swish inner + He, Tanh outer + Xavier)
    run_experiment('he', 'xavier', 1, 2, 'h=swish, o=tanh', 0.01)
    
    # 4. Pure Swish (to compare against the hybrid)
    try:
        run_experiment('he', 'he', 1, 0, 'h=swish, o=linear', 0.0005)
    except Exception as e:
        print(f"H-Init: he     | O-Init: he     | LR: 0.0100 | Act: h=swish, o=linear          | Crashed (Exploded)")