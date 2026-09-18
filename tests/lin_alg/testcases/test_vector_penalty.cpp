#include "test_lin_alg.h"

#include <limits>

TEST_CASE("Vector: Penalty scaling", "[vector],[operation],[penalty]")
{
  const OSQPInt n = 4;

  OSQPFloat a1_val[4] = {1.5, 2.5, 3.5, 4.5};
  OSQPFloat a2_val[4] = {0.5, 1.5, 0.0, 3.5};
  OSQPFloat d_val[4]  = {2.0, 4.0, 6.0, 8.0};
  OSQPFloat E_val[4]  = {0.5, 2.0, 8.0, 32.0};
  OSQPInt   type_val[4] = {OSQP_PENALTY_NONE,  OSQP_PENALTY_L1L2,
                           OSQP_PENALTY_HUBER, OSQP_PENALTY_L1L2};

  OSQPFloat c = 0.25;

  OSQPVectorf_ptr a1{OSQPVectorf_malloc(n)};
  OSQPVectorf_ptr a2{OSQPVectorf_malloc(n)};
  OSQPVectorf_ptr d{OSQPVectorf_malloc(n)};
  OSQPVectorf_ptr E{OSQPVectorf_malloc(n)};
  OSQPVectorf_ptr ref{OSQPVectorf_malloc(n)};
  OSQPVectori_ptr type{OSQPVectori_malloc(n)};

  OSQPVectorf_from_raw(E.get(), E_val);
  OSQPVectori_from_raw(type.get(), type_val);

  SECTION("Per-row types")
  {
    OSQPFloat ref_a1[4] = {1.5, 0.3125, 0.013671875, 0.03515625};
    OSQPFloat ref_a2[4] = {0.5, 0.09375, 0.0, 0.0008544921875};
    OSQPFloat ref_d[4] = {2.0, 4.0, 48.0, 8.0};

    OSQPVectorf_from_raw(a1.get(), a1_val);
    OSQPVectorf_from_raw(a2.get(), a2_val);
    OSQPVectorf_from_raw(d.get(),  d_val);

    OSQPVectorf_ew_scale_penalty(a1.get(), a2.get(), d.get(), type.get(),
                                 OSQP_PENALTY_NONE, c, E.get(), 0);

    OSQPVectorf_from_raw(ref.get(), ref_a1);
    mu_assert("Error scaling alpha1",
              OSQPVectorf_norm_inf_diff(ref.get(), a1.get()) < TESTS_TOL);

    OSQPVectorf_from_raw(ref.get(), ref_a2);
    mu_assert("Error scaling alpha2",
              OSQPVectorf_norm_inf_diff(ref.get(), a2.get()) < TESTS_TOL);

    OSQPVectorf_from_raw(ref.get(), ref_d);
    mu_assert("Error scaling delta",
              OSQPVectorf_norm_inf_diff(ref.get(), d.get()) < TESTS_TOL);
  }

  SECTION("A NULL type vector applies the default to every row")
  {
    OSQPVectorf_ptr b1{OSQPVectorf_malloc(n)};
    OSQPVectorf_ptr b2{OSQPVectorf_malloc(n)};
    OSQPVectorf_ptr bd{OSQPVectorf_malloc(n)};

    OSQPInt uniform[4] = {OSQP_PENALTY_HUBER, OSQP_PENALTY_HUBER,
                          OSQP_PENALTY_HUBER, OSQP_PENALTY_HUBER};
    OSQPVectori_ptr utype{OSQPVectori_malloc(n)};
    OSQPVectori_from_raw(utype.get(), uniform);

    OSQPVectorf_from_raw(a1.get(), a1_val);
    OSQPVectorf_from_raw(a2.get(), a2_val);
    OSQPVectorf_from_raw(d.get(),  d_val);
    OSQPVectorf_from_raw(b1.get(), a1_val);
    OSQPVectorf_from_raw(b2.get(), a2_val);
    OSQPVectorf_from_raw(bd.get(), d_val);

    OSQPVectorf_ew_scale_penalty(a1.get(), a2.get(), d.get(), OSQP_NULL,
                                 OSQP_PENALTY_HUBER, c, E.get(), 0);
    OSQPVectorf_ew_scale_penalty(b1.get(), b2.get(), bd.get(), utype.get(),
                                 OSQP_PENALTY_NONE, c, E.get(), 0);

    mu_assert("Default type differs from an explicit uniform vector",
              OSQPVectorf_norm_inf_diff(a1.get(), b1.get()) < TESTS_TOL);
    mu_assert("Default type differs from an explicit uniform vector",
              OSQPVectorf_norm_inf_diff(a2.get(), b2.get()) < TESTS_TOL);
    mu_assert("Default type differs from an explicit uniform vector",
              OSQPVectorf_norm_inf_diff(bd.get(), d.get()) < TESTS_TOL);
  }

  SECTION("Unscaling inverts scaling")
  {
    // A finite weight crosses the public sentinel when scaled; a true hard
    // weight remains distinguishable and both must return to caller units.
    a1_val[1] = OSQP_INFTY / 2;
    a1_val[3] = OSQP_INFTY;
    E_val[1] = 0.0625;
    OSQPVectorf_from_raw(E.get(), E_val);
    OSQPVectorf_from_raw(a1.get(), a1_val);
    OSQPVectorf_from_raw(a2.get(), a2_val);
    OSQPVectorf_from_raw(d.get(),  d_val);

    OSQPVectorf_ew_scale_penalty(a1.get(), a2.get(), d.get(), type.get(),
                                 OSQP_PENALTY_NONE, c, E.get(), 0);
    OSQPFloat scaled[4];
    OSQPVectorf_to_raw(scaled, a1.get());
    REQUIRE(scaled[1] == 2 * OSQP_INFTY);
    REQUIRE(scaled[3] == std::numeric_limits<OSQPFloat>::infinity());
    OSQPVectorf_ew_scale_penalty(a1.get(), a2.get(), d.get(), type.get(),
                                 OSQP_PENALTY_NONE, c, E.get(), 1);

    OSQPVectorf_from_raw(ref.get(), a1_val);
    mu_assert("Round trip changed alpha1",
              OSQPVectorf_norm_inf_diff(ref.get(), a1.get()) < TESTS_TOL);

    OSQPVectorf_from_raw(ref.get(), a2_val);
    mu_assert("Round trip changed alpha2",
              OSQPVectorf_norm_inf_diff(ref.get(), a2.get()) < TESTS_TOL);

    OSQPVectorf_from_raw(ref.get(), d_val);
    mu_assert("Round trip changed delta",
              OSQPVectorf_norm_inf_diff(ref.get(), d.get()) < TESTS_TOL);
  }

}
