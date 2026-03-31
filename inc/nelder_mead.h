/*
 *  Purpose:
 *
 *    NELMIN minimizes a function using the Nelder-Mead algorithm.
 *
 *  Discussion:
 *
 *    This routine seeks the minimum value of a user-specified function.
 *
 *    Simplex function minimisation procedure due to Nelder+Mead (1965), as
 *    implemented by O'Neill (1971, Appl.Statist. 20, 338-45), with subsequent
 *    comments by Chambers+Ertel (1974, 23, 250-1), Benyon (1976, 25, 97) and
 *    Hill (1978, 27, 380-2).
 *
 *    The function to be minimized must be defined by a function(al) of the
 *    form
 *
 *      real fn ( const std::array<real,n>& x )
 *
 *    where "real" can be any floating-point type, e.g. double.
 *
 *    This routine does not include a termination test using the fitting of a
 *    quadratic surface.
 *
 *  Licensing:
 *
 *    This code is distributed under the GNU LGPL license.
 *
 *  Author:
 *
 *    Original FORTRAN77 version by R O'Neill (1971)
 *    C version by John Burkardt (last modified 28 October 2010).
 *    Port to modern C++ by Piotr Różański. Numerical behaviour unchanged:
 *      - output variables moved to returned struct type
 *      - function is passed as std::function
 *      - floating-point type and number of variables are now template arguments
 *      - std::array is now used instead of pointers to raw arrays
 *      - overall comments and code formatting
 *
 *  Reference:
 *
 *    John Nelder, Roger Mead,
 *    A simplex method for function minimization,
 *    Computer Journal,
 *    Volume 7, 1965, pages 308-313.
 *
 *    R ONeill,
 *    Algorithm AS 47:
 *    Function Minimization Using a Simplex Procedure,
 *    Applied Statistics,
 *    Volume 20, Number 3, 1971, pages 338-345.
 *    https://doi.org/10.2307/2346772
 *
 *    Sonnet 4.6 generated this version using the source from:
 *    https://github.com/develancer/nelder-mead/
 */
#ifndef PTR_NELDER_MEAD_H
#define PTR_NELDER_MEAD_H

#include <array>
#include <climits>
#include <functional>

/**
 * Plain data object with output information from the run of nelder_mead routine.
 *
 * @tparam real floating-point type to be used, e.g. double
 * @tparam n the number of variables
 */
template<typename real, int n>
struct nelder_mead_result {
    std::array<real,n> xmin;
    real ynewlo;
    int icount;
    int numres;
    int ifault;
};

/**
 * This routine seeks the minimum value of a user-specified function.
 *
 * @tparam real floating-point type to be used, e.g. double
 * @tparam n the number of variables
 * @param fn the function to be minimized
 * @param start a starting point for the iteration
 * @param reqmin the terminating limit for the variance of function values
 * @param step determines the size and shape of the initial simplex;
 * the relative magnitudes of its elements should reflect the units of the variables
 * @param konvge the convergence check is carried out every konvge iterations
 * @param kcount the maximum number of function evaluations
 * @return structure with output information
 */
template<typename real, int n>
nelder_mead_result<real,n> nelder_mead(
    const std::function<real(const std::array<real,n> &)> &fn,
    std::array<real,n> start,
    real reqmin,
    const std::array<real,n> &step,
    int konvge = 1,
    int kcount = INT_MAX
)
{
    // Standard Nelder-Mead coefficients.
    const real contraction   = 0.5;
    const real expansion     = 2.0;
    const real reflection    = 1.0;
    const real factorial_eps = 0.001;  // perturbation size for local-minimum check

    nelder_mead_result<real,n> result;

    if (reqmin <= 0.0 || n < 1 || konvge < 1) {
        result.ifault = 1;
        return result;
    }

    std::array<std::array<real,n>, n + 1> p;
    std::array<real,n> pstar, p2star, pbar;
    std::array<real, n + 1> fval;

    result.icount = 0;
    result.numres = 0;

    int steps_until_check = konvge;
    real simplex_scale = 1.0;
    const real variance_threshold = reqmin * n;

    // Outer loop: re-entered on restart.
    while (true) {
        // Build simplex: one vertex at start, plus one per dimension shifted by step.
        p[n] = start;
        fval[n] = fn(start);
        result.icount++;

        for (int j = 0; j < n; j++) {
            real saved = start[j];
            start[j] += step[j] * simplex_scale;
            p[j] = start;
            fval[j] = fn(start);
            result.icount++;
            start[j] = saved;
        }

        // Identify the initial best vertex.
        real best_val = fval[0];
        int best = 0;
        for (int i = 1; i <= n; i++) {
            if (fval[i] < best_val) {
                best_val = fval[i];
                best = i;
            }
        }

        // Inner loop: one Nelder-Mead step per iteration.
        while (result.icount <= kcount) {
            // Find the worst (highest) vertex.
            result.ynewlo = fval[0];
            int worst = 0;
            for (int i = 1; i <= n; i++) {
                if (result.ynewlo < fval[i]) {
                    result.ynewlo = fval[i];
                    worst = i;
                }
            }

            // Centroid of all vertices except the worst.
            for (int i = 0; i < n; i++) {
                real sum = 0.0;
                for (int j = 0; j <= n; j++) {
                    sum += p[j][i];
                }
                pbar[i] = (sum - p[worst][i]) / n;
            }

            // Reflect worst through centroid.
            for (int i = 0; i < n; i++) {
                pstar[i] = pbar[i] + reflection * (pbar[i] - p[worst][i]);
            }
            real reflected_val = fn(pstar);
            result.icount++;

            if (reflected_val < best_val) {
                // Reflection beat the best — try extending further.
                for (int i = 0; i < n; i++) {
                    p2star[i] = pbar[i] + expansion * (pstar[i] - pbar[i]);
                }
                real expanded_val = fn(p2star);
                result.icount++;

                if (reflected_val < expanded_val) {
                    p[worst] = pstar;
                    fval[worst] = reflected_val;
                }
                else {
                    p[worst] = p2star;
                    fval[worst] = expanded_val;
                }
            }
            else {
                // Count how many vertices the reflection beats.
                int n_better = 0;
                for (int i = 0; i <= n; i++) {
                    if (reflected_val < fval[i]) {
                        n_better++;
                    }
                }

                if (n_better > 1) {
                    // Reflection is better than most — accept it.
                    p[worst] = pstar;
                    fval[worst] = reflected_val;
                }
                else if (n_better == 0) {
                    // Reflection is worse than everything — contract from the worst side.
                    for (int i = 0; i < n; i++) {
                        p2star[i] = pbar[i] + contraction * (p[worst][i] - pbar[i]);
                    }
                    real contracted_val = fn(p2star);
                    result.icount++;

                    if (fval[worst] < contracted_val) {
                        // Contraction didn't help — shrink the whole simplex toward best.
                        for (int j = 0; j <= n; j++) {
                            for (int i = 0; i < n; i++) {
                                p[j][i] = (p[j][i] + p[best][i]) * 0.5;
                            }
                            fval[j] = fn(p[j]);
                            result.icount++;
                        }
                        best_val = fval[0];
                        best = 0;
                        for (int i = 1; i <= n; i++) {
                            if (fval[i] < best_val) {
                                best_val = fval[i];
                                best = i;
                            }
                        }
                        continue;
                    }
                    else {
                        p[worst] = p2star;
                        fval[worst] = contracted_val;
                    }
                }
                else {
                    // n_better == 1: reflection only beat the worst — contract from reflection side.
                    for (int i = 0; i < n; i++) {
                        p2star[i] = pbar[i] + contraction * (pstar[i] - pbar[i]);
                    }
                    real contracted_val = fn(p2star);
                    result.icount++;

                    if (contracted_val <= reflected_val) {
                        p[worst] = p2star;
                        fval[worst] = contracted_val;
                    }
                    else {
                        p[worst] = pstar;
                        fval[worst] = reflected_val;
                    }
                }
            }

            // Update best if the replaced vertex improved.
            if (fval[worst] < best_val) {
                best_val = fval[worst];
                best = worst;
            }

            // Check convergence every konvge steps.
            if (--steps_until_check == 0) {
                steps_until_check = konvge;

                // Variance of function values via Welford's algorithm.
                real mean = 0.0, m2 = 0.0;
                for (int i = 0; i <= n; i++) {
                    real delta = fval[i] - mean;
                    mean += delta / (i + 1);
                    m2 += delta * (fval[i] - mean);
                }
                if (m2 <= variance_threshold) {
                    break;
                }
            }
        }

        // Record the best vertex found.
        result.xmin = p[best];
        result.ynewlo = fval[best];

        if (kcount < result.icount) {
            result.ifault = 2;
            break;
        }

        // Factorial test: verify xmin is a local minimum by perturbing each variable.
        result.ifault = 0;
        for (int i = 0; i < n; i++) {
            real perturb = step[i] * factorial_eps;

            result.xmin[i] += perturb;
            real fn_val = fn(result.xmin);
            result.icount++;
            if (fn_val < result.ynewlo) {
                result.ifault = 2;
                break;
            }

            result.xmin[i] -= 2.0 * perturb;
            fn_val = fn(result.xmin);
            result.icount++;
            if (fn_val < result.ynewlo) {
                result.ifault = 2;
                break;
            }

            result.xmin[i] += perturb;  // restore
        }

        if (result.ifault == 0) {
            break;
        }

        // Restart from the best point with a smaller simplex.
        start = result.xmin;
        simplex_scale = factorial_eps;
        result.numres++;
    }

    return result;
}

#endif // PTR_NELDER_MEAD_H
