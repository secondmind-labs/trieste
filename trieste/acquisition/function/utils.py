from typing import *

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


# =============================================================================
# Standard univariate normal CDF and inverse CDF for Multivariate Normal CDF
# =============================================================================

def standard_normal_cdf_and_inverse_cdf(dtype: tf.DType):
    """Returns two callables *Phi* and *iPhi*, which compute the cumulative
    density function and inverse cumulative density function of a standard
    univariate Gaussian.
    
    :param dtype: The data type to use, either tf.float32 or tf.float64.
    :returns Phi, iPhi: Cumulative and inverse cumulative density functions.
    """
    
    normal = tfp.distributions.Normal(
        loc=tf.zeros(shape=(), dtype=dtype),
        scale=tf.ones(shape=(), dtype=dtype),
    )
    Phi = lambda x: normal.cdf(x)
    iPhi = lambda x: normal.quantile(x)
    
    return Phi, iPhi


# =============================================================================
# Update index helper for Multivariate Normal CDF
# =============================================================================

def get_update_indices(B: int, S: int, Q: int, q: int):
    """Returns indices for updating a tensor using tf.tensor_scatter_nd_add,
    for use within the _mvn_cdf function, for computing the cumulative density
    function of a multivariate Gaussian. The indices *idx* returned are such
    that the following operation
        
        idx = get_update_indices(B, S, Q, q)
        tensor = tf.tensor_scatter_nd_add(tensor, idx, update)
        
    is equivalent to the numpy operation
        
        tensor = tensor[:, :, q] + update
        
    where *tensor* is a tensor of shape (B, S, Q).
    
    :param B: First dim. of tensor for which the indices are generated.
    :param S: Second dim. of tensor for which the indices are generated.
    :param Q: Third dim. of tensor for which the indices are generated.
    :param q: Index of tensor along fourth dim. to which the update is applied.
    """
    
    dtype = tf.int32
    
    idxB = tf.tile(tf.range(B, dtype=dtype)[:, None, None], (1, S, 1))
    idxS = tf.tile(tf.range(S, dtype=dtype)[None, :, None], (B, 1, 1))
    idxQ = tf.tile(tf.convert_to_tensor(q)[None, None, None], (B, S, 1))
    
    idx = tf.concat([idxB, idxS, idxQ], axis=-1)
    
    return idx
    

# =============================================================================
# Multivariate Normal CDF
# =============================================================================

def make_mvn_cdf(samples: tf.Tensor):
    """Builds the cumulative density function of the multivariate Gaussian 
    using the Genz approximation detailed in
    
        https://www.math.wsu.edu/faculty/genz/papers/mvn.pdf.
        
    This is a Monte Carlo approximation which is more accurate than a naive
    Monte Carlo estimate of the expected improvent. In order to use
    reparametrised samples, the helper accepts a tensor of samples, and the
    callable uses these fixed samples whenever it is called.
    
    :param samples: Tensor of shape (B, Q), with values between 0 and 1.
    :returns mvn_cdf: Function computing the MC approximation of the CDF.
    """
    
    # Unpack sample shape tensor
    S, Q = samples.shape

    @tf.function
    def mvn_cdf(
            x: tf.Tensor,
            mean: tf.Tensor,
            cov: tf.Tensor,
            jitter: float = 1e-6,
        ):
        """
        x: tf.Tensor, shape (B, Q)
        mean: tf.Tensor, shape (B, Q)
        cov: tf.Tensor, shape (B, Q, Q)
        """
        
        # Check shapes of input tensors
        tf.debugging.assert_shapes(
            [
                (x, ("B", "Q")),
                (mean, ("B", "Q")),
                (cov, ("B", "Q", "Q")),
            ]
        )

        # Identify data type to use for all calculations
        dtype = mean.dtype
        B, Q = mean.shape

        # Compute Cholesky factors
        jitter = jitter * tf.eye(Q, dtype=dtype)[None, :, :]
        C = tf.linalg.cholesky(cov + jitter)  # (B, Q, Q)

        # Rename samples and limits for brevity
        w = samples  # (S, Q)
        b = x - mean  # (B, Q)

        # Initialise transformation variables
        e = tf.zeros(shape=(B, S, Q), dtype=dtype)
        f = tf.zeros(shape=(B, S, Q), dtype=dtype)
        y = tf.zeros(shape=(B, S, Q), dtype=dtype)

        # Initialise standard normal for computing CDFs
        Phi, iPhi = standard_normal_cdf_and_inverse_cdf(dtype=dtype)

        # Get update indices for convenience later
        idx = get_update_indices(B=B, S=S, Q=Q, q=0)
            
        # Slice out common tensors
        b0 = b[:, None, 0]
        C0 = C[:, None, 0, 0] + 1e-12

        # Compute transformation variables at the first step
        e_update = tf.tile(Phi(b0 / C0), (1, S))  # (B, S)
        e = tf.tensor_scatter_nd_add(e, idx, e_update)
        f = tf.tensor_scatter_nd_add(f, idx, e_update)

        for i in tf.range(1, Q):

            # Update y tensor
            y_update = iPhi(
                1e-6 + (1 - 2e-6) * w[None, :, i-1] * e[:, :, i-1]
            )
            y = tf.tensor_scatter_nd_add(y, idx, y_update)
            
            # Slice out common tensors
            bi = b[:, None, i]
            Ci_ = C[:, None, i, :i]
            Cii = C[:, None, i, i] + 1e-12
            yi = y[:, :, :i]

            # Compute indices to update d, e and f tensors
            idx = get_update_indices(B=B, S=S, Q=Q, q=i)

            # Update e tensor
            e_update = Phi((bi - tf.reduce_sum(Ci_*yi, axis=-1)) / Cii)
            e = tf.tensor_scatter_nd_add(e, idx, e_update)

            # Update f tensor
            f_update = e[:, :, i] * f[:, :, i-1]
            f = tf.tensor_scatter_nd_add(f, idx, f_update)

        return tf.reduce_mean(f[:, :, -1], axis=-1)
    
    return mvn_cdf


@tf.function
def monte_carlo_expected_improvement(
        mean: tf.Tensor,
        covariance: tf.Tensor,
        threshold: tf.Tensor,
        num_samples: int = int(1e4),
    ):
    
    # Check shapes of covariance tensor
    tf.debugging.assert_shapes(
        [
            (mean, ("B", "Q")),
            (covariance, ("B", "Q", "Q")),
            (threshold, ("B",)),
        ]
    )
    
    samples = tfd.MultivariateNormalFullCovariance(
        loc=mean,
        covariance_matrix=covariance,
    ).sample(sample_shape=[num_samples])
    
    # Check shapes of covariance tensor
    tf.debugging.assert_shapes(
        [
            (mean, ("B", "Q")),
            (covariance, ("B", "Q", "Q")),
            (threshold, ("B",)),
            (samples, (num_samples, "B", "Q")),
        ]
    )
    
    ei = tf.math.maximum(samples - threshold[None, :, None], 0.)
    ei = tf.reduce_max(ei, axis=-1)
    ei = tf.reduce_mean(ei, axis=0)
    
    return ei
    
    
