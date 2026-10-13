"""
Neural Networks From Scratch: Forward and Backward

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - numerical_gradient
def numerical_gradient(f, x, eps=1e-5):
    grads = np.zeros_like(x)

    for idx in np.ndindex(x.shape):
        x_plus = x.copy()
        x_minus = x.copy()

        x_plus[idx] += eps
        x_minus[idx] -= eps
        grads[idx] = (f(x_plus) - f(x_minus)) / (2 * eps)

    return grads

# Step 2 - gradient_check
def gradient_check(analytic_grad, numeric_grad, tol=1e-5):

    # TODO: Return max relative error between analytic and numeric gradients.
    analytic_grad = np.asarray(analytic_grad, dtype=float)
    numeric_grad  = np.asarray(numeric_grad, dtype=float)
    nominator = np.abs(numeric_grad - analytic_grad)
    denominator = np.maximum(tol, np.maximum(np.abs(analytic_grad), np.abs(numeric_grad)))

    return float(np.max(nominator / denominator))

# Step 3 - make_dense
def make_dense(in_dim, out_dim, weight_init_fn):
    """Create a fully connected layer.

    Inputs:
      in_dim: int, input feature size
      out_dim: int, output feature size
      weight_init_fn: callable(in_dim, out_dim) -> (W, b)

    Returns layer dict with keys:
      params: {'W': (in_dim, out_dim), 'b': (out_dim,)}
      forward(x) -> (y, cache) with y shape (batch, out_dim)
      backward(dout, cache) -> (dx, grads) with grads {'W', 'b'}
        Analytic dx/dW/db must match numerical_gradient via gradient_check.
    """
    # TODO: your approach here
    W, b = weight_init_fn(in_dim, out_dim)
    forward = lambda x: (x @ W + b, {"x": x})

    def backward(dout, cache):
      x = cache['x']

      dx = dout @ W.T
      dW = x.T @ dout
      db = dout.sum(axis=0)

      grads = {"W": dW, "b": db}

      return dx, grads 


    return {
      "params": {"W": W, "b": b},
      "forward": forward,
      "backward": backward,
    }

# Step 4 - make_activation
def make_activation(kind='relu'):
    """Create a genuinely nonlinear elementwise activation layer.

    Args:
        kind: str nonlinearity name. Default 'relu' must implement ReLU
              (zero negatives, pass non-negatives). Other kinds optional.

    Returns:
        Layer dict with:
          forward(x) -> (y, cache)
            x, y: np.ndarray shape (batch, dim)
          backward(dout, cache) -> (dx, {})
            dout, dx: np.ndarray shape (batch, dim)
            param grad dict is always empty (no learnable params)

    Must be elementwise and non-affine; analytic dx must match
    numerical_gradient / gradient_check.
    """
    forward = lambda x: (np.maximum(x, 0), {"x": x})
    def backward(dout, cache):
      x = cache['x']
      dx = dout * (x>0)
      
      return dx, {}
    
    return {
      "params": {},
      "forward": forward,
      "backward": backward
    }

# Step 5 - initialize_weights
def initialize_weights(in_dim, out_dim, scheme='he'):
    """Return (W, b) for a dense layer.

    Inputs:
      in_dim: int fan-in
      out_dim: int fan-out
      scheme: str initialization family (default 'he')

    Returns:
      W: np.ndarray shape (in_dim, out_dim), finite, symmetry-breaking,
         scale stable with depth (fan-in dependent)
      b: np.ndarray shape (out_dim,), near zero
    """
    W = np.random.normal(loc=0, scale=np.sqrt(2/in_dim), size=(in_dim, out_dim))
    b = np.zeros(out_dim)
    return W, b

# Step 6 - make_loss (not yet solved)
# TODO: implement

# Step 7 - make_sequential (not yet solved)
# TODO: implement

# Step 8 - forward_backward (not yet solved)
# TODO: implement

# Step 9 - make_optimizer (not yet solved)
# TODO: implement

# Step 10 - train_step (not yet solved)
# TODO: implement

# Step 11 - train (not yet solved)
# TODO: implement

# Step 12 - design_network (not yet solved)
# TODO: implement

# Step 13 - improve_generalization (not yet solved)
# TODO: implement

