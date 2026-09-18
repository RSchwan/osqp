#include <math.h>
#include "osqp.h"
#include "penalty.h"
#include "algebra_vector.h"
#include "auxil.h"
#include "error.h"
#include "printing.h"

/***********************************************************
* Soft-constraint penalties                              * *
***********************************************************/

/* The per-row types, or OSQP_NULL when every row uses default_penalty_type.
 * The element-wise kernels take the fast path on OSQP_NULL. */
static const OSQPVectori* penalty_types(const OSQPPenaltyData* pen) {
  return pen->uniform ? OSQP_NULL : pen->type;
}

/* Weights are required to be finite, so the declared type is the whole story:
 * a row is soft if and only if its type is not OSQP_PENALTY_NONE. */
const OSQPVectori* penalty_row_types(const OSQPSolver* solver) {
  OSQPPenaltyData* pen = solver->work->data->penalty;

  return pen ? penalty_types(pen) : OSQP_NULL;
}

OSQPInt penalty_default_type(const OSQPSolver* solver) {
  OSQPPenaltyData* pen = solver->work->data->penalty;

  return pen ? pen->default_penalty_type : OSQP_PENALTY_NONE;
}

void penalty_project(OSQPSolver*        solver,
                     OSQPVectorf*       z,
                     const OSQPVectorf* v) {

  OSQPWorkspace*   work = solver->work;
  OSQPPenaltyData* pen  = work->data->penalty;

  if (!pen) {
    OSQPVectorf_ew_bound_vec(z, v, work->data->l, work->data->u);
    return;
  }

  OSQPVectorf_ew_prox_penalty(z, v, work->data->l, work->data->u,
                              solver->settings->rho_is_vec ? work->rho_vec : OSQP_NULL,
                              solver->settings->rho,
                              pen->alpha1, pen->alpha2, pen->delta,
                              penalty_types(pen), pen->default_penalty_type);
}

/* Phi(R(z)), the penalty contribution to the primal objective. Scaled like the
 * rest of the objective, since the scaled penalty is c*phi(s/E). */
OSQPFloat penalty_obj_value(const OSQPSolver*  solver,
                            const OSQPVectorf* z) {

  OSQPWorkspace*   work = solver->work;
  OSQPPenaltyData* pen  = work->data->penalty;

  if (!pen) return 0.0;

  return OSQPVectorf_penalty_value(z, work->data->l, work->data->u,
                                   pen->alpha1, pen->alpha2, pen->delta,
                                   penalty_types(pen), pen->default_penalty_type,
                                   work->penalty_val_tmp);
}

/* Phi*(y), the penalty contribution to the dual objective. OSQP_INFTY if y is
 * outside dom Phi*, which ADMM iterates never are but polish and warm starts
 * can be. */
OSQPFloat penalty_conj_value(const OSQPSolver*  solver,
                             const OSQPVectorf* y) {

  OSQPWorkspace*   work = solver->work;
  OSQPPenaltyData* pen  = work->data->penalty;

  if (!pen) return 0.0;

  return OSQPVectorf_penalty_conj_value(y, pen->alpha1, pen->alpha2, pen->delta,
                                        penalty_types(pen), pen->default_penalty_type,
                                        work->penalty_val_tmp);
}

#if OSQP_EMBEDDED_MODE != 1

/* Scale or unscale the penalty parameters in place, like l and u */
static void penalty_apply_scaling(OSQPSolver* solver,
                                  OSQPInt     invert) {

  OSQPWorkspace*   work = solver->work;
  OSQPPenaltyData* pen;

  if (!work || !work->data) return;

  pen = work->data->penalty;

  if (!pen || !solver->settings->scaling) return;

  OSQPVectorf_ew_scale_penalty(pen->alpha1, pen->alpha2, pen->delta,
                               penalty_types(pen), pen->default_penalty_type,
                               work->scaling->c, work->scaling->E, invert);
}

void penalty_scale(OSQPSolver* solver) {
  penalty_apply_scaling(solver, 0);
}

void penalty_unscale(OSQPSolver* solver) {
  penalty_apply_scaling(solver, 1);
}

#endif /* if OSQP_EMBEDDED_MODE != 1 */

#ifndef OSQP_EMBEDDED_MODE

void penalty_free(OSQPWorkspace* work) {

  OSQPPenaltyData* pen;

  if (!work) return;

  OSQPVectori_free(work->penalty_type_tmp);
  work->penalty_type_tmp = OSQP_NULL;
  OSQPVectori_free(work->penalty_flags_tmp);
  work->penalty_flags_tmp = OSQP_NULL;
  OSQPVectorf_free(work->penalty_val_tmp);
  work->penalty_val_tmp = OSQP_NULL;

  work->penalty_any_soft          = 0;
  work->penalty_any_linear_growth = 0;

  if (!work->data) return;

  pen = work->data->penalty;

  if (pen) {
    OSQPVectori_free(pen->type);
    OSQPVectorf_free(pen->alpha1);
    OSQPVectorf_free(pen->alpha2);
    OSQPVectorf_free(pen->delta);
    c_free(pen);
  }

  work->data->penalty = OSQP_NULL;
}

/* Allocate the penalty data. The weights are left uninitialized: every entry
 * point that creates or changes types also establishes valid weights. */
static OSQPInt penalty_alloc(OSQPWorkspace* work,
                             OSQPInt        m) {

  OSQPPenaltyData* pen = c_calloc(1, sizeof(OSQPPenaltyData));

  if (!pen) return 1;

  pen->uniform              = 1;
  pen->default_penalty_type = OSQP_PENALTY_NONE;

  /* Calloc so the types are a defined OSQP_PENALTY_NONE even while uniform */
  pen->type   = OSQPVectori_calloc(m);
  pen->alpha1 = OSQPVectorf_malloc(m);
  pen->alpha2 = OSQPVectorf_malloc(m);
  pen->delta  = OSQPVectorf_malloc(m);

  work->data->penalty    = pen;
  work->penalty_type_tmp = OSQPVectori_calloc(m);
  work->penalty_flags_tmp = OSQPVectori_calloc(1);
  work->penalty_val_tmp   = OSQPVectorf_malloc(1);

  if (!(pen->type) || !(pen->alpha1) || !(pen->alpha2) || !(pen->delta) ||
      !(work->penalty_type_tmp) || !(work->penalty_flags_tmp) ||
      !(work->penalty_val_tmp)) {
    penalty_free(work);
    return 1;
  }

  return 0;
}

/* NB: a switch rather than a range check, so adding a type lands here */
static OSQPInt penalty_type_is_valid(OSQPInt type) {

  switch (type) {
    case OSQP_PENALTY_NONE:
    case OSQP_PENALTY_L1L2:
    case OSQP_PENALTY_HUBER:
      return 1;

    default:
      return 0;
  }
}

/* NB: braces are required, c_eprint expands to several statements */
static void penalty_report_errors(OSQPInt flags) {

  if (flags & OSQP_PENALTY_ERR_NEGATIVE) {
    c_eprint("L1L2 penalty requires alpha1 >= 0 and alpha2 >= 0");
  }

  if (flags & OSQP_PENALTY_ERR_INFINITE) {
    c_eprint("penalty weights must be finite and below OSQP_INFTY; "
             "use OSQP_PENALTY_NONE for a hard constraint");
  }

  if (flags & OSQP_PENALTY_ERR_HUBER_W) {
    c_eprint("Huber penalty requires alpha1 > 0");
  }

  if (flags & OSQP_PENALTY_ERR_HUBER_D) {
    c_eprint("Huber penalty requires delta > 0");
  }
}

static void penalty_update_flags(OSQPWorkspace* work) {

  OSQPPenaltyData* pen = work->data->penalty;

  OSQPVectorf_penalty_flags(pen->alpha1, pen->alpha2,
                            penalty_types(pen), pen->default_penalty_type,
                            &work->penalty_any_soft,
                            &work->penalty_any_linear_growth, work->penalty_flags_tmp);
}

/* Validate a default type and an optional per-row type array */
static OSQPInt penalty_validate_types(OSQPInt        default_type,
                                      const OSQPInt* type,
                                      OSQPInt        m) {
  OSQPInt j;

  if (!penalty_type_is_valid(default_type)) {
    c_eprint("default penalty type %" OSQP_INT_FMT " not recognized", default_type);
    return 1;
  }

  if (type) {
    for (j = 0; j < m; j++) {
      if (!penalty_type_is_valid(type[j])) {
        c_eprint("penalty type %" OSQP_INT_FMT " not recognized (row %" OSQP_INT_FMT ")",
                 type[j], j);
        return 1;
      }
    }
  }

  return 0;
}

/* Commit a type layout. Does not allocate. */
static void penalty_commit_types(OSQPPenaltyData* pen,
                                 OSQPInt          default_type,
                                 const OSQPInt*   type) {

  pen->default_penalty_type = default_type;
  pen->uniform              = type ? 0 : 1;

  if (type) OSQPVectori_from_raw(pen->type, type);
}

/* Validate staged weights, in the caller's units, against a type layout */
static OSQPInt penalty_check_params(OSQPWorkspace*     work,
                                    const OSQPVectorf* alpha1,
                                    const OSQPVectorf* alpha2,
                                    const OSQPVectorf* delta,
                                    const OSQPVectori* type,
                                    OSQPInt            default_type) {

  OSQPInt flags = OSQPVectorf_penalty_params_check(alpha1, alpha2, delta,
                                                   type, default_type,
                                                   work->penalty_flags_tmp);

  if (flags) penalty_report_errors(flags);

  return flags ? 1 : 0;
}

/* Refresh state derived from the type layout. */
static OSQPInt penalty_finish_types(OSQPSolver* solver) {

  penalty_update_flags(solver->work);

  /* The rho classification depends on which rows are soft, and osqp_update_rho
   * reuses the cached constr_type, so it has to be recomputed here */
  if (solver->settings->rho_is_vec && update_rho_vec(solver))
    return osqp_error(OSQP_LINSYS_SOLVER_INIT_ERROR);

  return 0;
}

OSQPInt osqp_setup_penalty(OSQPSolver*      solver,
                           OSQPInt          default_type,
                           const OSQPInt*   type,
                           const OSQPFloat* alpha1,
                           const OSQPFloat* alpha2,
                           const OSQPFloat* delta) {

  OSQPWorkspace*   work;
  OSQPPenaltyData* pen;

  /* Check if workspace has been initialized */
  if (!solver || !solver->work || !solver->work->data)
    return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  work = solver->work;

  if (work->data->penalty) {
    c_eprint("penalty already set up; use osqp_update_penalty_types");
    return osqp_error(OSQP_DATA_VALIDATION_ERROR);
  }

  if (!alpha1 || !alpha2 || !delta) {
    c_eprint("osqp_setup_penalty requires alpha1, alpha2 and delta; "
             "a soft row is never left without weights");
    return osqp_error(OSQP_DATA_VALIDATION_ERROR);
  }

  if (penalty_validate_types(default_type, type, work->data->m))
    return osqp_error(OSQP_DATA_VALIDATION_ERROR);

  if (penalty_alloc(work, work->data->m))
    return osqp_error(OSQP_MEM_ALLOC_ERROR);

  pen = work->data->penalty;

  penalty_commit_types(pen, default_type, type);

  if (work->data->m > 0) {
    OSQPVectorf_from_raw(pen->alpha1, alpha1);
    OSQPVectorf_from_raw(pen->alpha2, alpha2);
    OSQPVectorf_from_raw(pen->delta,  delta);

    if (penalty_check_params(work, pen->alpha1, pen->alpha2, pen->delta,
                             penalty_types(pen), pen->default_penalty_type)) {
      penalty_free(work);
      return osqp_error(OSQP_DATA_VALIDATION_ERROR);
    }

    penalty_scale(solver);
  }

  return penalty_finish_types(solver);
}

OSQPInt osqp_update_penalty_types(OSQPSolver*    solver,
                                  OSQPInt        default_type,
                                  const OSQPInt* type) {

  OSQPWorkspace*   work;
  OSQPPenaltyData* pen;
  OSQPVectorf*     alpha1;
  OSQPVectorf*     alpha2;
  OSQPVectorf*     delta;

  /* Check if workspace has been initialized */
  if (!solver || !solver->work || !solver->work->data)
    return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  work = solver->work;
  pen  = work->data->penalty;

  if (!pen) {
    c_eprint("no penalty has been set up; call osqp_setup_penalty first");
    return osqp_error(OSQP_DATA_NOT_INITIALIZED);
  }

  if (penalty_validate_types(default_type, type, work->data->m))
    return osqp_error(OSQP_DATA_VALIDATION_ERROR);

  /* Stage parameters in user units because a type change can alter the power
   * of E applied to them. Rejected updates leave the stored values untouched. */
  if (work->data->m > 0) {
    if (type) OSQPVectori_from_raw(work->penalty_type_tmp, type);

    alpha1 = work->z_prev;
    alpha2 = work->delta_y;
    delta  = work->Ax;

    OSQPVectorf_copy(alpha1, pen->alpha1);
    OSQPVectorf_copy(alpha2, pen->alpha2);
    OSQPVectorf_copy(delta,  pen->delta);

    if (solver->settings->scaling)
      OSQPVectorf_ew_scale_penalty(alpha1, alpha2, delta,
                                   penalty_types(pen), pen->default_penalty_type,
                                   work->scaling->c, work->scaling->E, 1);

    if (penalty_check_params(work, alpha1, alpha2, delta,
                             type ? work->penalty_type_tmp : OSQP_NULL,
                             default_type))
      return osqp_error(OSQP_DATA_VALIDATION_ERROR);

    if (solver->settings->scaling)
      OSQPVectorf_ew_scale_penalty(alpha1, alpha2, delta,
                                   type ? work->penalty_type_tmp : OSQP_NULL,
                                   default_type,
                                   work->scaling->c, work->scaling->E, 0);
  }

  penalty_commit_types(pen, default_type, type);

  if (work->data->m > 0) {
    OSQPVectorf_copy(pen->alpha1, alpha1);
    OSQPVectorf_copy(pen->alpha2, alpha2);
    OSQPVectorf_copy(pen->delta,  delta);
  }

  return penalty_finish_types(solver);
}

OSQPInt osqp_update_penalty_params(OSQPSolver*      solver,
                                   const OSQPFloat* alpha1_new,
                                   const OSQPFloat* alpha2_new,
                                   const OSQPFloat* delta_new) {

  OSQPWorkspace*   work;
  OSQPPenaltyData* pen;
  OSQPVectorf*     alpha1;
  OSQPVectorf*     alpha2;
  OSQPVectorf*     delta;

  /* Check if workspace has been initialized */
  if (!solver || !solver->work || !solver->work->data)
    return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  work = solver->work;
  pen  = work->data->penalty;

  if (!pen) {
    c_eprint("no penalty has been set up; call osqp_setup_penalty first");
    return osqp_error(OSQP_DATA_NOT_INITIALIZED);
  }

  if (work->data->m == 0) return 0;

  /* Assemble the candidate in workspace vectors that only carry state during a
   * solve, the same way osqp_update_data_vec borrows z_prev and delta_y */
  alpha1 = work->z_prev;
  alpha2 = work->delta_y;
  delta  = work->Ax;

  OSQPVectorf_copy(alpha1, pen->alpha1);
  OSQPVectorf_copy(alpha2, pen->alpha2);
  OSQPVectorf_copy(delta,  pen->delta);

  // Merge the new values over the current ones in the caller's units
  if (solver->settings->scaling)
    OSQPVectorf_ew_scale_penalty(alpha1, alpha2, delta,
                                 penalty_types(pen), pen->default_penalty_type,
                                 work->scaling->c, work->scaling->E, 1);

  if (alpha1_new) OSQPVectorf_from_raw(alpha1, alpha1_new);
  if (alpha2_new) OSQPVectorf_from_raw(alpha2, alpha2_new);
  if (delta_new)  OSQPVectorf_from_raw(delta,  delta_new);

  /* Validate before committing, so a rejected call leaves the penalty alone */
  if (penalty_check_params(work, alpha1, alpha2, delta,
                           penalty_types(pen), pen->default_penalty_type))
    return osqp_error(OSQP_DATA_VALIDATION_ERROR);

  if (solver->settings->scaling)
    OSQPVectorf_ew_scale_penalty(alpha1, alpha2, delta,
                                 penalty_types(pen), pen->default_penalty_type,
                                 work->scaling->c, work->scaling->E, 0);

  OSQPVectorf_copy(pen->alpha1, alpha1);
  OSQPVectorf_copy(pen->alpha2, alpha2);
  OSQPVectorf_copy(pen->delta,  delta);

  penalty_update_flags(work);
  return 0;
}

#endif /* ifndef OSQP_EMBEDDED_MODE */
