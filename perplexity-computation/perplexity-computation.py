def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    N = len(prob_distributions)
    if N == 0:
        return 0

    H = -(1 / N) * sum(
        math.log(max(prob_distributions[i][actual_tokens[i]], 1e-12)) 
        for i in range(N)
    )

    return math.exp(H)
        