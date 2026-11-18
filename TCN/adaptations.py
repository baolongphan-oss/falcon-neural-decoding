import numpy as np
from sklearn.metrics import r2_score

LAMBDAS = (0.1, 1, 10, 100, 1000, 10000)
SPLIT = 0.7

def anchored_refit(H, Y, W_old, b_old, lam):
    """
    H: (n, 64) features, Y: (n, 16) targets in standardized units
    W_old: head weights transposed (64, 16), b_old: (16,)
    """
    # add a column of ones to H, stack b_old under W_old  -> handles the bias
    col_of_ones = np.ones((len(H), 1))
    # add the column of ones at the far right
    H1 = np.hstack([H, col_of_ones]) # gives shape (n, 65)
    # add the bias as a row under W_old
    W1_old = np.vstack([W_old, b_old])

    # W = solve(HᵀH + λI, HᵀY + λW_old)
    # A: how the features relate to each other, 
    # plus λI = stiffness (bigger λ resists change more)
    A = H1.T @ H1 + lam * np.identity(H1.shape[1]) # shape (65, 65)
    # B: the pulls on W. HᵀY pulls toward fitting the new day; 
    # λW_old pulls back to the old head
    B = H1.T @ Y + lam * W1_old # shape (65, 16)
    # find the W that balances both pulls 
    # solve A·W = B
    W1 = np.linalg.solve(A, B)

    # return the new weights and bias
    return W1[:-1], W1[-1]

def choose_lambda(H, Y, W_old, b_old, lams=LAMBDAS):
    """
    Pick λ from [0.1, 1, 10, 100, 1000, 10000]:

    Fit on the first 70% of the calibration rows and score on the last 30%. 
    Split by time, not randomly, because neighboring bins look almost the same after smoothing, 
    so a random split would leak information.
    Keep the λ with the best R², 
    returns the best λ; caller then refits on all calibration rows.
    """
    cut = int(SPLIT * len(H))
    H_fit, Y_fit = H[:cut], Y[:cut]
    H_check, Y_check = H[cut:], Y[cut:]

    best_r2 = -float('inf')
    best_lambda = None
    for lambda_val in lams:
        W, b = anchored_refit(H_fit, Y_fit, W_old, b_old, lambda_val)
        pred = H_check @ W + b 
        r2 = r2_score(Y_check, pred, multioutput="variance_weighted")
        if r2 > best_r2:
            best_r2 = r2 
            best_lambda = lambda_val
    return best_lambda
    
