#ifndef PENALTY_H
#define PENALTY_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

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
