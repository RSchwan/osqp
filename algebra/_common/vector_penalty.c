#include <math.h>
#include "osqp.h"
#include "algebra_vector.h"
#include "algebra_impl.h"

/*******************************************************************************
 * Soft-constraint penalty vector operations                                   *
 *                                                                             *
 * Branchy element-wise work with no BLAS equivalent, so the builtin and MKL    *
 * backends share one implementation rather than each carrying a copy.          *
 *******************************************************************************/

#if OSQP_EMBEDDED_MODE != 1

/* Convert the public infinity sentinel at the scaling boundary. Finite scaled
 * weights may exceed OSQP_INFTY and must survive the inverse transform. */
static OSQPFloat ew_scale_penalty_weight(OSQPFloat weight,
                                         OSQPFloat factor,
                                         OSQPInt invert) {
  if (invert)
    return weight == (OSQPFloat)HUGE_VAL ? OSQP_INFTY : weight / factor;

  return weight >= OSQP_INFTY ? (OSQPFloat)HUGE_VAL : weight * factor;
}
void OSQPVectorf_ew_scale_penalty(OSQPVectorf*       alpha1,
                                  OSQPVectorf*       alpha2,
                                  OSQPVectorf*       delta,
                                  const OSQPVectori* type,
                                  OSQPInt            default_type,
                                  OSQPFloat          c,
                                  const OSQPVectorf* E,
                                  OSQPInt            invert) {

  OSQPInt    i;
  OSQPInt    length = alpha1->length;
  OSQPFloat* a1     = alpha1->values;
  OSQPFloat* a2     = alpha2->values;
  OSQPFloat* d      = delta->values;
  OSQPFloat* Ev     = E ? E->values : OSQP_NULL;
  OSQPInt*   tv     = type ? type->values : OSQP_NULL;

  for (i = 0; i < length; i++) {
    OSQPFloat e = Ev ? Ev[i] : 1.0;
    OSQPFloat f1, f2, fd;

    switch (tv ? tv[i] : default_type) {
      case OSQP_PENALTY_L1L2:
        f1 = c / e;       f2 = c / (e * e); fd = 1.0;
        break;

      case OSQP_PENALTY_HUBER:
        f1 = c / (e * e); f2 = 1.0;         fd = e;
        break;

      default:
        continue;   // Hard row, parameters unused
    }

    a1[i] = ew_scale_penalty_weight(a1[i], f1, invert);
    a2[i] = ew_scale_penalty_weight(a2[i], f2, invert);
    d[i]  = ew_scale_penalty_weight(d[i],  fd, invert);
  }
}
void OSQPVectorf_ew_reset_changed_penalty(OSQPVectorf*       alpha1,
                                          OSQPVectorf*       alpha2,
                                          OSQPVectorf*       delta,
                                          const OSQPVectori* old_type,
                                          OSQPInt            old_default,
                                          const OSQPVectori* new_type,
                                          OSQPInt            new_default) {

  OSQPInt    i;
  OSQPInt    length = alpha1->length;
  OSQPFloat* a1     = alpha1->values;
  OSQPFloat* a2     = alpha2->values;
  OSQPFloat* d      = delta->values;
  OSQPInt*   ot     = old_type ? old_type->values : OSQP_NULL;
  OSQPInt*   nt     = new_type ? new_type->values : OSQP_NULL;

  for (i = 0; i < length; i++) {
    OSQPInt told = ot ? ot[i] : old_default;
    OSQPInt tnew = nt ? nt[i] : new_default;

    if (told != tnew) {
      a1[i] = (OSQPFloat)HUGE_VAL;
      a2[i] = (OSQPFloat)HUGE_VAL;
      d[i]  = (OSQPFloat)HUGE_VAL;
    }
  }
}
OSQPInt OSQPVectorf_penalty_params_check(const OSQPVectorf* alpha1,
                                         const OSQPVectorf* alpha2,
                                         const OSQPVectorf* delta,
                                         const OSQPVectori* type,
                                         OSQPInt            default_type,
                                         OSQPVectori*       scratch) {

  OSQPInt    i;
  OSQPInt    flags  = 0;
  OSQPInt    length = alpha1->length;
  OSQPFloat* a1     = alpha1->values;
  OSQPFloat* a2     = alpha2->values;
  OSQPFloat* d      = delta->values;
  OSQPInt*   tv     = type ? type->values : OSQP_NULL;

  (void)scratch; /* Only device backends need reduction storage. */

  /* NB: tested as !(x >= 0) rather than (x < 0) so that NaN is rejected too */
  for (i = 0; i < length; i++) {
    switch (tv ? tv[i] : default_type) {
      case OSQP_PENALTY_L1L2:
        if (!(a1[i] >= 0.0) || !(a2[i] >= 0.0))      flags |= OSQP_PENALTY_ERR_NEGATIVE;
        else if ((a1[i] <= 0.0) && (a2[i] <= 0.0))   flags |= OSQP_PENALTY_ERR_ZERO;
        break;

      case OSQP_PENALTY_HUBER:
        if (!(a1[i] > 0.0)) flags |= OSQP_PENALTY_ERR_HUBER_W;
        if (!(d[i]  > 0.0)) flags |= OSQP_PENALTY_ERR_HUBER_D;
        break;

      default:
        break;
    }
  }

  return flags;
}
void OSQPVectorf_penalty_flags(const OSQPVectorf* alpha2,
                               const OSQPVectori* type,
                               OSQPInt            default_type,
                               OSQPInt*           any_soft,
                               OSQPInt*           any_linear_growth,
                               OSQPVectori*       scratch) {

  OSQPInt    i;
  OSQPInt    length = alpha2->length;
  OSQPFloat* a2     = alpha2->values;
  OSQPInt*   tv     = type ? type->values : OSQP_NULL;

  (void)scratch; /* Only device backends need reduction storage. */

  *any_soft          = 0;
  *any_linear_growth = 0;

  /* NB: growth follows from alpha2, not the type: an L1L2 row with alpha2 == 0
   * is a pure L1 penalty and grows linearly, alpha2 > 0 makes it superlinear.
   * Scaling factors are positive, so the test works on scaled values too. */
  for (i = 0; i < length; i++) {
    switch (tv ? tv[i] : default_type) {
      case OSQP_PENALTY_L1L2:
        *any_soft = 1;
        if (a2[i] <= 0.0) *any_linear_growth = 1;
        break;

      case OSQP_PENALTY_HUBER:
        *any_soft          = 1;
        *any_linear_growth = 1;
        break;

      default:
        break;
    }
  }
}

#endif /* OSQP_EMBEDDED_MODE != 1 */
