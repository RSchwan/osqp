#ifndef PENALTY_H
#define PENALTY_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Generalized projection z = vbar + prox_{Phi/rho}(v - vbar) of the z-update,
 * where vbar = Proj_[l,u](v). Reduces to the plain projection when no penalty
 * is set up. It is acceptable to assign z == v.
 * @param solver Solver
 * @param z      Output vector
 * @param v      Input vector
 */
void penalty_project(OSQPSolver*        solver,
                     OSQPVectorf*       z,
                     const OSQPVectorf* v);

/**
 * Penalty part of the primal objective, Phi(R(z)), on the scaled data.
 * Zero when no penalty is set up.
 * @param  solver Solver
 * @param  z      Constraint iterate
 */
OSQPFloat penalty_obj_value(const OSQPSolver*  solver,
                            const OSQPVectorf* z);

/**
 * Penalty part of the dual objective, Phi*(y), on the scaled data.
 * OSQP_INFTY if y lies outside dom Phi*, zero when no penalty is set up.
 * @param  solver Solver
 * @param  y      Dual iterate
 */
OSQPFloat penalty_conj_value(const OSQPSolver*  solver,
                             const OSQPVectorf* y);

/**
 * Per-row penalty types, or OSQP_NULL when every row uses the default.
 * @param  solver Solver
 */
const OSQPVectori* penalty_row_types(const OSQPSolver* solver);

/**
 * Penalty type of the rows not covered by penalty_row_types()
 * @param  solver Solver
 */
OSQPInt penalty_default_type(const OSQPSolver* solver);

# if OSQP_EMBEDDED_MODE != 1

/**
 * Scale/unscale the penalty parameters in place, like l and u.
 * Called from scale_data/unscale_data; no-ops if no penalty is allocated.
 * @param  solver Solver
 */
void penalty_scale(OSQPSolver* solver);
void penalty_unscale(OSQPSolver* solver);

# endif /* if OSQP_EMBEDDED_MODE != 1 */

# ifndef OSQP_EMBEDDED_MODE

/**
 * Free the penalty data and its derived workspace state
 * @param work Workspace
 */
void penalty_free(OSQPWorkspace* work);

# endif /* ifndef OSQP_EMBEDDED_MODE */

#ifdef __cplusplus
}
#endif

#endif /* ifndef PENALTY_H */
