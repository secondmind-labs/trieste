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




# =============================================================================
# Helper for computing b and m tensors for qEI objective
# =============================================================================

def compute_bm(
        mean: tf.Tensor,
        threshold: tf.Tensor,
    ):
    """Helper function for the batch expected improvement, which computes the
    tensors b and m as detailed in Chevallier and Ginsbourger
    
        https://hal.archives-ouvertes.fr/hal-00732512v2/document.
    
    :param mean: Tensor of shape (B, Q)
    :param threshold: Tensor of shape (B,)
        
    :returns b: Tensor of shape (B, Q, Q)
    :returns m: Tensor of shape (B, Q, Q)
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
    """Helper function for the compute_Sigma function, which computes a *delta*
    tensor of shape (B, idx, idx) such that
    
        delta[B, i, :] = 1 if i == idx
        delta[B, i, :] = 0 otherwise.
        
    If transpose == True, then the last two dimensions of the tensor are
    transposed, in which case
    
        delta[B, :, i] = 1 if i == idx
        delta[B, :, i] = 0 otherwise.
        
    :param idx: Index for entries equal to 1.
    :param dim: Dimension of the last and second to last axes.
    :param B: Leading dimension of tensor.
    :param transpose: Whether to transpose the last two dimensions or not.
    :param dtype: The dtype of the tensor, either tf.float32 or tf.float64.
    """
        
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
    """Helper function for the batch expected improvement, which computes the
    tensor Sigma, as detailed in Chevallier and Ginsbourger
    
        https://hal.archives-ouvertes.fr/hal-00732512v2/document.
    
    :param covariance: Tensor of shape (B, Q, Q)
    :returns Sigma: Tensor of shape (B, Q, Q, Q)
    """
        
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
    """Helper function for the batch expected improvement, which computes the
    tensor p, as detailed in Chevallier and Ginsbourger
    
        https://hal.archives-ouvertes.fr/hal-00732512v2/document.
    
    :param m_reshaped: Tensor of shape (BQ, Q)
    :param b_reshaped: Tensor of shape (BQ, Q)
    :param Sigma_reshaped: Tensor of shape (BQ, Q, Q)
    :returns p: Tensor of shape (B, Q)
    """
    
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
    """Helper function for the batch expected improvement, which computes the
    tensor c, which is the c^{(i)} tensor detailed in Chevallier and
    Ginsbourger
    
        https://hal.archives-ouvertes.fr/hal-00732512v2/document.
    
    :param m_reshaped: Tensor of shape (BQ, Q)
    :param b_reshaped: Tensor of shape (BQ, Q)
    :param Sigma_reshaped: Tensor of shape (BQ, Q, Q)
    :returns c: Tensor of shape (B, Q, Q-1)
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
# Helper for R tensor for qEI objective
# =============================================================================

def compute_R(Sigma_reshaped: tf.Tensor):
    """Helper function for the batch expected improvement, which computes the
    tensor R, which is the Sigma^{(i)} tensor detailed in Chevallier an
    Ginsbourger
    
        https://hal.archives-ouvertes.fr/hal-00732512v2/document.
    
    :param Sigma_reshaped: Tensor of shape (BQ, Q, Q)
    :returns R: Tensor of shape (B, Q-1, Q-1)
    """
    
    # Check shapes of covariance tensor
    tf.debugging.assert_shapes([(Sigma_reshaped, ("BQ", "Q", "Q"))])
    
    # Unpack tensor shape
    BQ, Q, _ = Sigma_reshaped.shape
    
    Sigma_uv = tf.tile(Sigma_reshaped[:, None, :, :], (1, Q, 1, 1))
    Sigma_iu = tf.tile(Sigma_reshaped[:, :, :, None], (1, 1, 1, Q))
    Sigma_iv = tf.tile(Sigma_reshaped[:, :, None, :], (1, 1, Q, 1))
    Sigma_ii = tf.linalg.diag_part(Sigma_reshaped)[:, :, None, None]
    
    R_whole = Sigma_uv - Sigma_iu * Sigma_iv / Sigma_ii
    
    def create_blocks(q):
        
        block1 = tf.concat(
            [
                R_whole[:, q, :q, :q],
                R_whole[:, q, q+1:, :q],
            ],
            axis=1,
        )
        
        block2 = tf.concat(
            [
                R_whole[:, q, :q, q+1:],
                R_whole[:, q, q+1:, q+1:],
            ],
            axis=1,
        )
        
        R_block = tf.concat([block1, block2], axis=2)
        
        return R_block
    
    R = tf.map_fn(
        create_blocks,
        tf.range(Q),
        fn_output_signature=R_whole.dtype,
    )
    R = tf.transpose(R, perm=[1, 0, 2, 3])
    
    return R


# =============================================================================
# Helper for Phi tensor for qEI objective
# =============================================================================

def compute_Phi(
        c: tf.Tensor,
        R: tf.Tensor,
        mvn_cdf: Callable,
    ):
    """Helper function for the batch expected improvement, which computes the
    tensor Phi, which is the tensor of multivariate Gaussian CDFs, in the inner
    sum of the equation (3) in Chevallier and Ginsbourger
    
        https://hal.archives-ouvertes.fr/hal-00732512v2/document.
    
    :param c: Tensor of shape (BQ, Q, Q-1).
    :param R: Tensor of shape (BQ, Q, Q-1, Q-1).
    :param mvn_cdf: Multivariate Gaussian CDF, made using make_mvn_cdf.
    :returns Phi: Tensor of multivariate Gaussian CDFs.
    """
    
    # Check shapes of covariance tensor
    tf.debugging.assert_shapes(
        [
            (c, ("BQ", "Q", "Q_")),
            (R, ("BQ", "Q", "Q_", "Q_")),
        ]
    )
    
    # Unpack tensor shape and data type
    BQ, Q, _, Q_ = R.shape
    dtype = R.dtype
    
    if BQ % Q == 0:
        B = BQ // Q
        
    else:
        raise ValueError(
            f"Expected size of dimension 0 of R tensor to be "
            f"divisible by size of dimension 1, instead found "
            f"{R.shape[0]} and {R.shape[1]}."
        )

    c_reshaped = tf.reshape(c, (BQ*Q, Q-1))
    R_reshaped = tf.reshape(R, (BQ*Q, Q-1, Q-1))
    
    # Compute mean, covariance and x for Phi mvn normal cdf
    Phi_cdf_x = c_reshaped  # (B*Q, Q-1)
    Phi_cdf_mean = tf.zeros(shape=(BQ*Q, Q-1), dtype=dtype)  # (B*Q*Q, Q)
    Phi_cdf_cov = R_reshaped  # (B*Q*Q, Q-1, Q-1)
    
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

def qEI(
        mean: tf.Tensor,
        covariance: tf.Tensor,
        threshold: tf.Tensor,
        mvn_cdf: Callable,
    ):
    """Accurate Monte Carlo approximation of the batch expected improvement,
    using the method of Chevallier and Ginsbourger
    
        https://hal.archives-ouvertes.fr/hal-00732512v2/document.
    
    :param mean: Tensor of shape (B, Q).
    :param covariance: Tensor of shape (B, Q, Q).
    :param threshold: Tensor of shape (B, Q).
    :param mvn_cdf: Callable computing the multivariate CDF of a Gaussian.
    :returns qei: Tensor of shape (B,), the expected improvement.
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
    R = compute_R(
        Sigma_reshaped=Sigma_reshaped,
    ) # (B*Q, Q, Q-1, Q-1)
    
    # Compute Q-1 multivariate CDFs
    Phi_mvn_cdfs = compute_Phi(
        c=c,
        R=R,
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
    qei = tf.reduce_sum(mean_T_term + sum_term, axis=1)
    
    return qei


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
    
    
