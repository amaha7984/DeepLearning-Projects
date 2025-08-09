# Problem 123: Calculate Computational Efficiency of MoE

def compute_efficiency(n_experts, k_active, d_in, d_out):
    """
    Calculate computational savings of MoE vs. dense layer.

    Args:
        n_experts: Total number of experts
        k_active: Number of active experts (sparsity)
        d_in: Input dimension
        d_out: Output dimension

    Returns:
        Percentage savings in FLOPs
    """
    dense = n_experts * d_in * d_out
    moe = k_active * d_in * d_out
     
    efficiency = ((dense - moe) / dense) * 100
    return efficiency
