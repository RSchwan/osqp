#include <math.h>
#include "osqp.h"
#include "algebra_vector.h"
#include "algebra_impl.h"

/*******************************************************************************
 * Soft-constraint penalty vector operations                                   *
 *                                                                             *
 * Branchy element-wise work with no BLAS equivalent, so the builtin and MKL    *
 * backends share one implementation rather than each carrying a copy.          *
 * Uniform soft penalties deliberately use the same loop as mixed penalties;   *
 * only uniform hard penalties bypass it. CUDA specializes uniform kernels.    *
 *******************************************************************************/

/* prox_{phi/rho}(r) for a single row */
static OSQPFloat ew_prox_penalty_row(OSQPFloat r,
                                     OSQPFloat rho,
                                     OSQPInt   type,
                                     OSQPFloat a1,
                                     OSQPFloat a2,
                                     OSQPFloat d) {

  OSQPFloat t;

  switch (type) {
    case OSQP_PENALTY_L1L2:
      /* rho/(rho + alpha2) * S_{alpha1/rho}(r) */
      t = c_absval(r) - a1 / rho;
      if (t <= 0.0) return 0.0;
      return (r > 0.0 ? t : -t) * rho / (rho + a2);

    case OSQP_PENALTY_HUBER:
      /* Test the quadratic candidate instead of forming (1 + a1/rho)*d,
         whose intermediate ratio can overflow for finite scaled weights.
         Divide r first for large weights, so rho/(rho+a1) cannot underflow;
         otherwise form the bounded ratio first to avoid enlarging r. */
      t = a1 > 1.0 ? (r / (rho + a1)) * rho
                   : (rho / (rho + a1)) * r;
      if (c_absval(t) <= d) return t;
      /* Divide first only when rho reduces the product. */
      t = rho >= 1.0 ? (a1 / rho) * d : (a1 * d) / rho;
      return r - t * (r > 0.0 ? 1.0 : -1.0);

    default:
      return 0.0;   // Hard row
  }
}

/* phi(s) for a single row */
static OSQPFloat penalty_value_row(OSQPFloat s,
                                   OSQPInt   type,
                                   OSQPFloat a1,
                                   OSQPFloat a2,
                                   OSQPFloat d) {

  OSQPFloat as = c_absval(s);

  /* NB: s is exactly zero on a hard row and on any row whose weight is
     infinite, so returning here also avoids forming 0 * inf */
  if (as == 0.0) return 0.0;

  switch (type) {
    case OSQP_PENALTY_L1L2:
      return a1 * as + 0.5 * a2 * s * s;

    case OSQP_PENALTY_HUBER:
      return as <= d ? 0.5 * a1 * s * s : a1 * (d * as - 0.5 * d * d);

    default:
      return 0.0;
  }
}

/* phi*(y) for a single row, OSQP_INFTY outside the domain */
static OSQPFloat penalty_conj_row(OSQPFloat y,
                                  OSQPInt   type,
                                  OSQPFloat a1,
                                  OSQPFloat a2,
                                  OSQPFloat d) {

  OSQPFloat ay = c_absval(y);
  OSQPFloat t;

  switch (type) {
    case OSQP_PENALTY_L1L2:
      /* Pure L1 has an indicator conjugate; an infinite quadratic weight
         has a zero conjugate. Handle both before any division or square. */
      if (a2 <= 0.0) return ay <= a1 ? 0.0 : OSQP_INFTY;
      t = ay - a1;
      return t <= 0.0 ? 0.0 : (0.5 * t) * (t / a2);

    case OSQP_PENALTY_HUBER:
      return ay <= a1 * d ? (0.5 * ay) * (ay / a1) : OSQP_INFTY;

    default:
      return 0.0;   // Hard row, phi* = 0
  }
}

void OSQPVectorf_ew_prox_penalty(OSQPVectorf*       z,
                                 const OSQPVectorf* v,
                                 const OSQPVectorf* l,
                                 const OSQPVectorf* u,
                                 const OSQPVectorf* rho_vec,
                                 OSQPFloat          rho,
                                 const OSQPVectorf* alpha1,
                                 const OSQPVectorf* alpha2,
                                 const OSQPVectorf* delta,
                                 const OSQPVectori* type,
                                 OSQPInt            default_type) {

  OSQPInt    i;
  OSQPInt    length = z->length;
  OSQPFloat* zv     = z->values;
  OSQPFloat* vv     = v->values;
  OSQPFloat* lv     = l->values;
  OSQPFloat* uv     = u->values;
  OSQPFloat* a1     = alpha1->values;
  OSQPFloat* a2     = alpha2->values;
  OSQPFloat* d      = delta->values;
  OSQPFloat* rv     = rho_vec ? rho_vec->values : OSQP_NULL;
  OSQPInt*   tv     = type ? type->values : OSQP_NULL;

  /* The all-hard case is the plain projection, bit for bit */
  if (!tv && (default_type == OSQP_PENALTY_NONE)) {
    OSQPVectorf_ew_bound_vec(z, v, l, u);
    return;
  }

  for (i = 0; i < length; i++) {
    OSQPFloat vbar = c_min(c_max(vv[i], lv[i]), uv[i]);

    zv[i] = vbar + ew_prox_penalty_row(vv[i] - vbar, rv ? rv[i] : rho,
                                       tv ? tv[i] : default_type,
                                       a1[i], a2[i], d[i]);
  }
}

OSQPFloat OSQPVectorf_penalty_value(const OSQPVectorf* z,
                                    const OSQPVectorf* l,
                                    const OSQPVectorf* u,
                                    const OSQPVectorf* alpha1,
                                    const OSQPVectorf* alpha2,
                                    const OSQPVectorf* delta,
                                    const OSQPVectori* type,
                                    OSQPInt            default_type,
                                    OSQPVectorf*       scratch) {

  OSQPInt    i;
  OSQPFloat  val    = 0.0;
  OSQPInt    length = z->length;
  OSQPFloat* zv     = z->values;
  OSQPFloat* lv     = l->values;
  OSQPFloat* uv     = u->values;
  OSQPFloat* a1     = alpha1->values;
  OSQPFloat* a2     = alpha2->values;
  OSQPFloat* d      = delta->values;
  OSQPInt*   tv     = type ? type->values : OSQP_NULL;

  (void)scratch; /* Only device backends need reduction storage. */

  if (!tv && (default_type == OSQP_PENALTY_NONE)) return 0.0;

  for (i = 0; i < length; i++) {
    OSQPFloat zbar = c_min(c_max(zv[i], lv[i]), uv[i]);

    val += penalty_value_row(zv[i] - zbar, tv ? tv[i] : default_type,
                             a1[i], a2[i], d[i]);
  }

  return val;
}

/* Recession rate of one row along w. Hard and superlinear rows return infinity
 * when the direction leaves the bound recession cone. */
static OSQPFloat penalty_reccone_rate_row(OSQPFloat w,
                                          OSQPFloat l,
                                          OSQPFloat u,
                                          OSQPInt   type,
                                          OSQPFloat a1,
                                          OSQPFloat a2,
                                          OSQPFloat d,
                                          OSQPFloat infval,
                                          OSQPFloat tol) {

  OSQPFloat slope;
  OSQPFloat dist;

  /* Distance from rec[l,u]. Linear penalties charge the exact distance;
     tol only relaxes the membership test for blocking rows. */
  if ((u < +infval) && (w > 0.0))      dist = w;
  else if ((l > -infval) && (w < 0.0)) dist = -w;
  else                                 return 0.0;

  switch (type) {
    case OSQP_PENALTY_L1L2:
      if (a2 > 0.0) return dist > tol ? OSQP_INFTY : 0.0;
      slope = a1;
      break;

    case OSQP_PENALTY_HUBER:
      /* NB: the asymptotic slope of alpha1*h_delta is alpha1*delta */
      slope = a1 * d;
      break;

    default:
      return dist > tol ? OSQP_INFTY : 0.0;
  }

  return slope * dist;
}

OSQPFloat OSQPVectorf_penalty_reccone_rate(const OSQPVectorf* w,
                                           const OSQPVectorf* l,
                                           const OSQPVectorf* u,
                                           const OSQPVectorf* alpha1,
                                           const OSQPVectorf* alpha2,
                                           const OSQPVectorf* delta,
                                           const OSQPVectori* type,
                                           OSQPInt            default_type,
                                           OSQPFloat          infval,
                                           OSQPFloat          tol,
                                           OSQPVectorf*       scratch) {

  OSQPInt    i;
  OSQPFloat  rate   = 0.0;
  OSQPInt    length = w->length;
  OSQPFloat* wv     = w->values;
  OSQPFloat* lv     = l->values;
  OSQPFloat* uv     = u->values;
  OSQPFloat* a1     = alpha1->values;
  OSQPFloat* a2     = alpha2->values;
  OSQPFloat* d      = delta->values;
  OSQPInt*   tv     = type ? type->values : OSQP_NULL;

  (void)scratch; /* Only device backends need reduction storage. */

  for (i = 0; i < length; i++) {
    OSQPFloat row_rate = penalty_reccone_rate_row(wv[i], lv[i], uv[i],
                                                  tv ? tv[i] : default_type,
                                                  a1[i], a2[i], d[i], infval, tol);

    if (row_rate >= OSQP_INFTY) return OSQP_INFTY;
    rate += row_rate;
  }

  return rate;
}

OSQPFloat OSQPVectorf_penalty_conj_value(const OSQPVectorf* y,
                                         const OSQPVectorf* alpha1,
                                         const OSQPVectorf* alpha2,
                                         const OSQPVectorf* delta,
                                         const OSQPVectori* type,
                                         OSQPInt            default_type,
                                         OSQPVectorf*       scratch) {

  OSQPInt    i;
  OSQPFloat  val    = 0.0;
  OSQPInt    length = y->length;
  OSQPFloat* yv     = y->values;
  OSQPFloat* a1     = alpha1->values;
  OSQPFloat* a2     = alpha2->values;
  OSQPFloat* d      = delta->values;
  OSQPInt*   tv     = type ? type->values : OSQP_NULL;

  (void)scratch; /* Only device backends need reduction storage. */

  if (!tv && (default_type == OSQP_PENALTY_NONE)) return 0.0;

  for (i = 0; i < length; i++) {
    OSQPFloat c = penalty_conj_row(yv[i], tv ? tv[i] : default_type,
                                   a1[i], a2[i], d[i]);

    if (c >= OSQP_INFTY) return OSQP_INFTY;

    val += c;
  }

  return val;
}

#if OSQP_EMBEDDED_MODE != 1

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

    if (invert) {
      a1[i] /= f1;  a2[i] /= f2;  d[i] /= fd;
    }
    else {
      a1[i] *= f1;  a2[i] *= f2;  d[i] *= fd;
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
        if (!(a1[i] >= 0.0) || !(a2[i] >= 0.0))          flags |= OSQP_PENALTY_ERR_NEGATIVE;
        if ((a1[i] >= OSQP_INFTY) || (a2[i] >= OSQP_INFTY)) flags |= OSQP_PENALTY_ERR_INFINITE;
        break;

      case OSQP_PENALTY_HUBER:
        if (!(a1[i] > 0.0)) flags |= OSQP_PENALTY_ERR_HUBER_W;
        if (!(d[i]  > 0.0)) flags |= OSQP_PENALTY_ERR_HUBER_D;
        if ((a1[i] >= OSQP_INFTY) || (d[i] >= OSQP_INFTY)) flags |= OSQP_PENALTY_ERR_INFINITE;
        break;

      default:
        break;
    }
  }

  return flags;
}
#endif /* OSQP_EMBEDDED_MODE != 1 */
