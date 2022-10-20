from typing import *

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions



# =============================================================================
# Vector-dot helper for Multivariate Normal CDF
# =============================================================================

def vector_dot(a: tf.Tensor, b: tf.Tensor):
    return tf.reduce_sum(a*b, axis=-1)


# =============================================================================
# Standard univariate normal CDF and inverse CDF for Multivariate Normal CDF
# =============================================================================

def standard_normal_cdf_and_inverse_cdf(dtype: tf.DType):
    
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
    
    # Unpack sample shape tensor
    S, Q = samples.shape

    @tf.function
    def _mvn_cdf(
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
            e_update = Phi((bi - vector_dot(Ci_, yi)) / Cii)
            e = tf.tensor_scatter_nd_add(e, idx, e_update)

            # Update f tensor
            f_update = e[:, :, i] * f[:, :, i-1]
            f = tf.tensor_scatter_nd_add(f, idx, f_update)

        return tf.reduce_mean(f[:, :, -1], axis=-1)
    
    return _mvn_cdf




# =============================================================================
# Helper for computing b and m tensors for qEI objective
# =============================================================================

def compute_bm(
        mean: tf.Tensor,
        threshold: tf.Tensor,
    ):
    """
    Arguments:
        mean: tf.Tensor, shape (B, Q)
        threshold: tf.Tensor, shape (B,)
        
    Returns:
        b: tf.Tensor, shape (B, Q, Q), b[B, K, Q]
        m: tf.Tensor, shape (B, Q, Q), m[B, K, Q]
    """
        
    # Check shapes of input tensors
    tf.debugging.assert_shapes(
        [
            (mean, ("B", "Q")),
            (threshold, ("B",)),
        ]
    )
    
    # Unpack tensor shape and data type
    B, Q = mean.shape
    dtype = mean.dtype
    
    # Compute b tensor
    threshold = tf.tile(threshold[:, None], (1, Q))
    threshold = tf.linalg.diag(threshold) # (B, Q, Q)
    
    b = tf.zeros(shape=(B, Q, Q), dtype=dtype)
    b = b - threshold
    
    # Compute m tensor
    m = mean[:, None, :] - mean[:, :, None]  # (B, Q, Q)
    m = m - tf.linalg.diag(mean)  # (B, Q, Q)
    
    return b, m


# =============================================================================
# Helper for masking tensors for qEI objective
# =============================================================================

def delta(
        idx: int,
        dim: int,
        B: int,
        transpose: bool,
        dtype: tf.DType
    ):
        
    # Check input parameters
    tf.debugging.assert_non_negative(idx)
    tf.debugging.assert_non_negative(dim)
    tf.debugging.assert_positive(B)
    
    o1 = tf.ones(shape=(B, idx, dim), dtype=dtype)
    z1 = tf.zeros(shape=(B, 1, dim), dtype=dtype)
    o2 = tf.ones(shape=(B, dim-idx-1, dim), dtype=dtype)
    
    delta = tf.concat([o1, z1, o2], axis=1)
    delta = tf.transpose(delta, perm=[0, 2, 1]) if transpose else delta
    
    return delta
    

# =============================================================================
# Helper for Sigma tensor for qEI objective
# =============================================================================

def compute_Sigma(covariance: tf.Tensor):
        
    # Check shapes of covariance tensor
    tf.debugging.assert_shapes([(covariance, ("B", "Q", "Q"))])
    
    # Unpack tensor shape and dtype
    B, Q, _ = covariance.shape
    dtype = covariance.dtype
    
    Sigma = tf.zeros(shape=(B, Q, Q, Q))
    
    def compute_single_slice(q):
        
        diq = delta(q, Q, B, transpose=False, dtype=dtype)
        dqj = delta(q, Q, B, transpose=True, dtype=dtype)
        
        Sigma_ij = covariance[:, :, :]
        Sigma_iq = covariance[:, :, q:q+1]
        Sigma_qj = covariance[:, q:q+1, :]
        Sigma_qq = covariance[:, q:q+1, q:q+1]
        
        cov = Sigma_ij * diq * dqj - Sigma_iq * diq - Sigma_qj * dqj + Sigma_qq
        
        return cov
    
    Sigma = tf.map_fn(
        compute_single_slice,
        tf.range(Q),
        fn_output_signature=dtype,
    )
    
    Sigma = tf.transpose(Sigma, perm=[1, 0, 2, 3])
        
    return Sigma


# =============================================================================
# Helper for p tensor (multivariate CDF) for qEI objective
# =============================================================================

def compute_p(
        m_reshaped: tf.Tensor,
        b_reshaped: tf.Tensor,
        Sigma_reshaped: tf.Tensor,
        mvn_cdf: Callable,
    ):
    
    # Check shapes of covariance tensor
    tf.debugging.assert_shapes(
        [
            (m_reshaped, ("BQ", "Q")),
            (b_reshaped, ("BQ", "Q")),
            (Sigma_reshaped, ("BQ", "Q", "Q")),
        ]
    )
    
    # Unpack dtype and mean shape
    dtype = m_reshaped.dtype
    BQ, Q = m_reshaped.shape # (B*Q, Q)
    
    if BQ % Q == 0:
        B = BQ // Q
        
    else:
        raise ValueError(
            f"Expected size of dimension 0 of m_reshaped tensor to be "
            f"divisible by size of dimension 1, instead found "
            f"{m_reshaped.shape[0]} and {m_reshaped.shape[1]}."
        )
    
    # Compute mean, covariance and x for p mvn normal cdf
    p_cdf_mean = tf.zeros(shape=(BQ, Q), dtype=dtype)  # (B*Q, Q)
    p_cdf_cov = Sigma_reshaped # (B*Q, Q, Q)
    
    p_cdf_x = b_reshaped - m_reshaped  # (B*Q, Q)
    
    p = mvn_cdf(
        x=p_cdf_x,
        mean=p_cdf_mean,
        cov=p_cdf_cov,
    )  # (B*Q,)
    
    p = tf.reshape(p, shape=(B, Q))  # (B, Q)
    
    return p


# =============================================================================
# Helper for c tensor for qEI objective
# =============================================================================

def compute_c(
        m_reshaped: tf.Tensor,
        b_reshaped: tf.Tensor,
        Sigma_reshaped: tf.Tensor,
    ):
    """
    Arguments:
        m_reshaped: tf.Tensor, shape (B*Q, Q)
        b_reshaped: tf.Tensor, shape (B*Q, Q)
        Sigma_reshaped: tf.Tensor, shape (B*Q, Q, Q)
    """
    
    # Check shapes of covariance tensor
    tf.debugging.assert_shapes(
        [
            (m_reshaped, ("BQ", "Q")),
            (b_reshaped, ("BQ", "Q")),
            (Sigma_reshaped, ("BQ", "Q", "Q")),
        ]
    )
    
    # Unpack tensor shape and data type
    BQ, Q = m_reshaped.shape
    dtype = m_reshaped.dtype
    
    # Compute difference between b and m tensors
    diff = b_reshaped - m_reshaped # (B*Q, Q)
    
    # Compute c, including the ith entry, which we want to remove
    cov_ratio = Sigma_reshaped / tf.linalg.diag_part(
        Sigma_reshaped
    )[:, :, None] # (B*Q, Q, Q)
    c = diff[:, None, :] - diff[:, :, None] * cov_ratio # (B*Q, Q, Q)
    
    # Remove the ith entry by masking c with a boolean mask with False across
    # the diagonal and True in the off-diagonal terms
    mask = tf.math.logical_not(
        tf.cast(tf.eye(Q, dtype=tf.int32), dtype=tf.bool)
    )
    mask = tf.tile(mask[None, :, :], (c.shape[0], 1, 1))
    
    c = tf.ragged.boolean_mask(c, mask).to_tensor()
    
    return c


# =============================================================================
# Helper for Sigmai tensor for qEI objective
# =============================================================================

def compute_Sigmai_matrix(Sigma_reshaped: tf.Tensor):
    """
    Sigma_reshaped: tf.Tensor, shape (B*Q, Q, Q)
    """
    
    # Check shapes of covariance tensor
    tf.debugging.assert_shapes([(Sigma_reshaped, ("BQ", "Q", "Q"))])
    
    # Unpack tensor shape
    BQ, Q, _ = Sigma_reshaped.shape
    
    Sigma_uv = tf.tile(Sigma_reshaped[:, None, :, :], (1, Q, 1, 1))
    Sigma_iu = tf.tile(Sigma_reshaped[:, :, :, None], (1, 1, 1, Q))
    Sigma_iv = tf.tile(Sigma_reshaped[:, :, None, :], (1, 1, Q, 1))
    Sigma_ii = tf.linalg.diag_part(Sigma_reshaped)[:, :, None, None]
    
    Sigmai_whole = Sigma_uv - Sigma_iu * Sigma_iv / Sigma_ii
    
    def create_blocks(q):
        
        block1 = tf.concat(
            [
                Sigmai_whole[:, q, :q, :q],
                Sigmai_whole[:, q, q+1:, :q],
            ],
            axis=1,
        )
        
        block2 = tf.concat(
            [
                Sigmai_whole[:, q, :q, q+1:],
                Sigmai_whole[:, q, q+1:, q+1:],
            ],
            axis=1,
        )
        
        Sigmai_block = tf.concat([block1, block2], axis=2)
        
        return Sigmai_block
    
    Sigmai = tf.map_fn(
        create_blocks,
        tf.range(Q),
        fn_output_signature=Sigmai_whole.dtype,
    )
    Sigmai = tf.transpose(Sigmai, perm=[1, 0, 2, 3])
    
    return Sigmai


# =============================================================================
# Helper for Phi tensor for qEI objective
# =============================================================================

def compute_Phi(
        c: tf.Tensor,
        Sigmai: tf.Tensor,
        mvn_cdf: Callable,
    ):
    """
    Arguments:
        c: tf.Tensor, shape (B*Q, Q, Q-1)
        Sigmai: tf.Tensor, shape (B*Q, Q, Q-1, Q-1)
    """
    
    # Check shapes of covariance tensor
    tf.debugging.assert_shapes(
        [
            (c, ("BQ", "Q", "Q_")),
            (Sigmai, ("BQ", "Q", "Q_", "Q_")),
        ]
    )
    
    # Unpack tensor shape and data type
    BQ, Q, _, Q_ = Sigmai.shape
    dtype = Sigmai.dtype
    
    if BQ % Q == 0:
        B = BQ // Q
        
    else:
        raise ValueError(
            f"Expected size of dimension 0 of Sigmai tensor to be "
            f"divisible by size of dimension 1, instead found "
            f"{Sigmai.shape[0]} and {Sigmai.shape[1]}."
        )

    c_reshaped = tf.reshape(c, (BQ*Q, Q-1))
    Sigmai_reshaped = tf.reshape(Sigmai, (BQ*Q, Q-1, Q-1))
    
    # Compute mean, covariance and x for Phi mvn normal cdf
    Phi_cdf_x = c_reshaped  # (B*Q, Q-1)
    Phi_cdf_mean = tf.zeros(shape=(BQ*Q, Q-1), dtype=dtype)  # (B*Q*Q, Q)
    Phi_cdf_cov = Sigmai_reshaped  # (B*Q*Q, Q-1, Q-1)
    
    # Compute multivariate cdfs
    mvn_cdfs = mvn_cdf(
        x=Phi_cdf_x,
        mean=Phi_cdf_mean,
        cov=Phi_cdf_cov,
    )
    mvn_cdfs = tf.reshape(mvn_cdfs, (B, Q, Q))  # (B, Q, Q)
    
    return mvn_cdfs


# =============================================================================
# The qEI objective
# =============================================================================

# @tf.function
def qEI(
        mean: tf.Tensor,
        covariance: tf.Tensor,
        threshold: tf.Tensor,
        mvn_cdf: Callable,
    ):
    """
    Arguments:
        mean: tf.Tensor, shape (B, Q)
        covariance: tf.Tensor, shape (B, Q, Q)
        threshold: tf.Tensor, shape (B,)
    """
    
    # Check shapes of covariance tensor
    tf.debugging.assert_shapes(
        [
            (mean, ("B", "Q")),
            (covariance, ("B", "Q", "Q")),
            (threshold, ("B",)),
        ]
    )
    
    # Unpack dtype and mean shape
    dtype = mean.dtype
    B, Q = mean.shape
    
    # Compute b and m tensors
    b, m = compute_bm(
        mean=mean,
        threshold=threshold,
    ) # (B, Q, Q), (B, Q, Q)
        
    # Compute Sigma
    Sigma = compute_Sigma(
        covariance=covariance
    ) # (B, Q, Q, Q)
    
    # Reshape all tensors, for batching
    b_reshaped = tf.reshape(b, (B*Q, Q))
    m_reshaped = tf.reshape(m, (B*Q, Q))
    Sigma_reshaped = tf.reshape(Sigma, (B*Q, Q, Q))
    
    # Compute p tensor
    p = compute_p(
        m_reshaped=m_reshaped,
        b_reshaped=b_reshaped,
        Sigma_reshaped=Sigma_reshaped,
        mvn_cdf=mvn_cdf,
    )
        
    # Compute c
    c = compute_c(
        m_reshaped=m_reshaped,
        b_reshaped=b_reshaped,
        Sigma_reshaped=Sigma_reshaped,
    ) # (B*Q, Q, Q-1)
        
    # Compute Sigma_i
    Sigmai = compute_Sigmai_matrix(
        Sigma_reshaped=Sigma_reshaped,
    ) # (B*Q, Q, Q-1, Q-1)
    
    # Compute Q-1 multivariate CDFs
    Phi_mvn_cdfs = compute_Phi(
        c=c,
        Sigmai=Sigmai,
        mvn_cdf=mvn_cdf,
    )
        
    # Compute univariate pdfs
    S_diag = tf.linalg.diag_part(Sigma)
    normal = tfp.distributions.Normal(loc=m, scale=S_diag**0.5)
    uvn_pdfs = tf.math.exp(normal.log_prob(b))  # (B, Q, Q)
    
    Sigma_diag = tf.linalg.diag_part(
        tf.transpose(Sigma, perm=[0, 2, 1, 3])
    )
    Sigma_diag = tf.transpose(Sigma_diag, perm=[0, 2, 1])
    
    T = tf.tile(threshold[:, None], (1, Q))
    
    mean_T_term = (mean - T) * p
    
    # Compute inner sum
    sum_term = tf.reduce_sum(
        Sigma_diag * uvn_pdfs * Phi_mvn_cdfs,
        axis=2,
    )
    
    # Compute outer sum
    qEI = tf.reduce_sum(mean_T_term + sum_term, axis=1)
    mcei = monte_carlo_expected_improvement(
        mean=mean,
        covariance=covariance,
        threshold=threshold,
    )
    
    # print(f"{tf.reduce_mean(qEI):.10f} {tf.reduce_mean(mcei):.10f}")
    
    return qEI


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
    
    
