#include <math.h>
#include "osqp.h"
#include "penalty.h"
#include "algebra_vector.h"
#include "error.h"
#include "printing.h"

/***********************************************************
* Soft-constraint penalties                              * *
***********************************************************/

#if OSQP_EMBEDDED_MODE != 1

/* The per-row types, or OSQP_NULL when every row uses default_penalty_type.
 * The element-wise kernels take the fast path on OSQP_NULL. */
static const OSQPVectori* penalty_types(const OSQPPenaltyData* pen) {
  return pen->uniform ? OSQP_NULL : pen->type;
}

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

/* Allocate the penalty data with all rows hard */
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

  if (!(pen->type) || !(pen->alpha1) || !(pen->alpha2) || !(pen->delta) ||
      !(work->penalty_type_tmp) || !(work->penalty_flags_tmp)) {
    penalty_free(work);
    return 1;
  }

  /* An infinite weight collapses every prox to zero, so a row stays hard
   * until the caller supplies weights */
  OSQPVectorf_set_scalar(pen->alpha1, (OSQPFloat)HUGE_VAL);
  OSQPVectorf_set_scalar(pen->alpha2, (OSQPFloat)HUGE_VAL);
  OSQPVectorf_set_scalar(pen->delta,  (OSQPFloat)HUGE_VAL);

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

  if (flags & OSQP_PENALTY_ERR_ZERO) {
    c_eprint("L1L2 penalty requires alpha1 > 0 or alpha2 > 0; "
             "a zero penalty would drop the constraint entirely");
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

  OSQPVectorf_penalty_flags(pen->alpha2, penalty_types(pen), pen->default_penalty_type,
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

OSQPInt osqp_setup_penalty(OSQPSolver*    solver,
                           OSQPInt        default_type,
                           const OSQPInt* type) {

  OSQPWorkspace* work;

  /* Check if workspace has been initialized */
  if (!solver || !solver->work || !solver->work->data)
    return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  work = solver->work;

  if (work->data->penalty) {
    c_eprint("penalty already set up; use osqp_update_penalty_types");
    return osqp_error(OSQP_DATA_VALIDATION_ERROR);
  }

  if (penalty_validate_types(default_type, type, work->data->m))
    return osqp_error(OSQP_DATA_VALIDATION_ERROR);

  if (penalty_alloc(work, work->data->m))
    return osqp_error(OSQP_MEM_ALLOC_ERROR);

  /* penalty_alloc leaves every parameter infinite, so the rows declared soft
   * here still behave as hard until osqp_update_penalty_params is called */
  penalty_commit_types(work->data->penalty, default_type, type);

  penalty_update_flags(work);

  return 0;
}

OSQPInt osqp_update_penalty_types(OSQPSolver*    solver,
                                  OSQPInt        default_type,
                                  const OSQPInt* type) {

  OSQPWorkspace*   work;
  OSQPPenaltyData* pen;
  OSQPInt          old_default;

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

  old_default = pen->default_penalty_type;

  /* Changing a row's type discards its weights, which were validated against
   * the old type. The row stays hard until new weights are supplied.
   * NB: the incoming types go to the staging vector rather than pen->type,
   * which is still needed here as the previous state. */
  if (type) OSQPVectori_from_raw(work->penalty_type_tmp, type);

  OSQPVectorf_ew_reset_changed_penalty(pen->alpha1, pen->alpha2, pen->delta,
                                       penalty_types(pen), old_default,
                                       type ? work->penalty_type_tmp : OSQP_NULL,
                                       default_type);

  penalty_commit_types(pen, default_type, type);

  penalty_update_flags(work);

  return 0;
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
  OSQPInt          m;
  OSQPInt          flags;

  /* Check if workspace has been initialized */
  if (!solver || !solver->work || !solver->work->data)
    return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  work = solver->work;
  pen  = work->data->penalty;
  m    = work->data->m;

  if (!pen) {
    c_eprint("no penalty has been set up; call osqp_setup_penalty first");
    return osqp_error(OSQP_DATA_NOT_INITIALIZED);
  }

  if (m == 0) return 0;

  /* Assemble the candidate in workspace vectors that only carry state during a
   * solve, the same way osqp_update_data_vec borrows z_prev and delta_y */
  alpha1 = work->z_prev;
  alpha2 = work->delta_y;
  delta  = work->Ax;

  OSQPVectorf_copy(alpha1, pen->alpha1);
  OSQPVectorf_copy(alpha2, pen->alpha2);
  OSQPVectorf_copy(delta,  pen->delta);

  // Merge the new values over the current ones in the caller's units
  OSQPVectorf_ew_scale_penalty(alpha1, alpha2, delta,
                               penalty_types(pen), pen->default_penalty_type,
                               solver->settings->scaling ? work->scaling->c : 1.0,
                               solver->settings->scaling ? work->scaling->E : OSQP_NULL,
                               1);

  if (alpha1_new) OSQPVectorf_from_raw(alpha1, alpha1_new);
  if (alpha2_new) OSQPVectorf_from_raw(alpha2, alpha2_new);
  if (delta_new)  OSQPVectorf_from_raw(delta,  delta_new);

  /* Validate before committing, so a rejected call leaves the penalty alone */
  flags = OSQPVectorf_penalty_params_check(alpha1, alpha2, delta,
                                           penalty_types(pen),
                                           pen->default_penalty_type,
                                           work->penalty_flags_tmp);

  if (flags) {
    penalty_report_errors(flags);
    return osqp_error(OSQP_DATA_VALIDATION_ERROR);
  }

  OSQPVectorf_ew_scale_penalty(alpha1, alpha2, delta,
                               penalty_types(pen), pen->default_penalty_type,
                               solver->settings->scaling ? work->scaling->c : 1.0,
                               solver->settings->scaling ? work->scaling->E : OSQP_NULL,
                               0);

  OSQPVectorf_copy(pen->alpha1, alpha1);
  OSQPVectorf_copy(pen->alpha2, alpha2);
  OSQPVectorf_copy(pen->delta,  delta);

  penalty_update_flags(work);

  return 0;
}

#endif /* ifndef OSQP_EMBEDDED_MODE */
