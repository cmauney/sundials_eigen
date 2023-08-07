#pragma once

#include <map>

#include <cvode/cvode.h> /* prototypes for CVODE fcts., consts.  */
#include <fmt/format.h>
#include <nvector/nvector_serial.h>  /* access to serial N_Vector            */
#include <sundials/sundials_types.h> /* defs. of realtype, sunindextype      */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver      */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix            */

// #include <Eigen/Core>
// #include <Eigen/Dense>

#include "types.hh"
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
// #define Ith(v,i)    NV_Ith_S(v,i-1)         /* Ith numbers components 1..NEQ
// */ #define IJth(A,i,j) SM_ELEMENT_D(A,i-1,j-1) /* IJth numbers
// rows,cols 1..NEQ */

using namespace types;

namespace cvode_wrapper {

inline std::map<std::string_view, int> cv_lmm = {{"BDF", CV_BDF},
                                                 {"ADAMS", CV_ADAMS}};

struct cv_options {
  // Integrator settings
  std::string step_method = "BDF";
  double rtol = 1.0E-8;  // relative tolerance
  double atol = 1.0E-8;  // absolute tolerance
  int max_steps = 500;   // max number of steps between outputs
  int max_order = 5;     // max order of linear multistep
  double t0 = 0.0;       // inital time
  double tf = 1.0;       // final time
  double step_min = 0.0; // minimum step size
  double step_max = 0.0; // maximum step size (0 -> no max size)

  // Linear solver and preconditioner settings
  double epslin = 1.0E-6; // linear solver tolerance factor

  // output options
  int output_lvl = 0;
};

static int check_retval(void *returnvalue, const char *funcname, int opt) {
  int *retval;

  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
  if (opt == 0 && returnvalue == NULL) {
    fmt::print(stderr,
               "\nSUNDIALS_ERROR: {} failed - returned NULL pointer\n\n",
               funcname);
    return (1);
  }

  /* Check if retval < 0 */
  else if (opt == 1) {
    retval = (int *)returnvalue;
    if (*retval < 0) {
      fmt::print(stderr, "\nSUNDIALS_ERROR: {} failed with retval = %d\n\n",
                 funcname, *retval);
      return (1);
    }
  }

  /* Check if function returned NULL pointer - no memory allocated */
  else if (opt == 2 && returnvalue == NULL) {
    fmt::print(stderr, "\nMEMORY_ERROR: {} failed - returned NULL pointer\n\n",
               funcname);
    return (1);
  }

  return (0);
}

/* compare the solution at the final time 4e10s to a reference solution computed
   using a relative tolerance of 1e-8 and absoltue tolerance of 1e-14 */
static int check_ans(N_Vector y, realtype t, realtype rtol, N_Vector atol) {
  int passfail = 0; /* answer pass (0) or fail (1) retval */
  N_Vector ref;     /* reference solution vector        */
  N_Vector ewt;     /* error weight vector              */
  realtype err;     /* wrms error                       */
  realtype ONE = RCONST(1.0);
  realtype ZERO = RCONST(0.0);

  /* create reference solution and error weight vectors */
  ref = N_VClone(y);
  ewt = N_VClone(y);

  /* set the reference solution data */
  NV_Ith_S(ref, 0) = RCONST(5.2083495894337328e-08);
  NV_Ith_S(ref, 1) = RCONST(2.0833399429795671e-13);
  NV_Ith_S(ref, 2) = RCONST(9.9999994791629776e-01);

  /* compute the error weight vector, loosen atol */
  N_VAbs(ref, ewt);
  N_VLinearSum(rtol, ewt, RCONST(10.0), atol, ewt);
  if (N_VMin(ewt) <= ZERO) {
    fmt::print(stderr, "\nSUNDIALS_ERROR: check_ans failed - ewt <= 0\n\n");
    return (-1);
  }
  fmt::print(stdout, "\n{} {} {}\n", NV_Ith_S(ewt, 0), NV_Ith_S(ewt, 1),
             NV_Ith_S(ewt, 2));
  N_VInv(ewt, ewt);

  /* compute the solution error */
  N_VLinearSum(ONE, y, -ONE, ref, ref);
  err = N_VWrmsNorm(ref, ewt);

  /* is the solution within the tolerances? */
  passfail = (err < ONE) ? 0 : 1;

  if (passfail) {
    fmt::print(stdout, "\nSUNDIALS_WARNING: check_ans error={}\n\n", err);
  }

  /* Free vectors */
  N_VDestroy(ref);
  N_VDestroy(ewt);

  return (passfail);
}

template <class System> struct cvode_stepper {
  // options
  cv_options options;

  // cvode context pointer
  void *cvode_mem;

  size_t N;            // system size
  N_Vector cv_y, abst; // ongoing solution
  SUNMatrix A;         //
  SUNLinearSolver LS;

  sundials::Context sunctx;

  cvode_stepper(cv_options opts)
      : options(std::move(opts)), cvode_mem(nullptr) {}

  ~cvode_stepper() {
    CVodeFree(&cvode_mem);
    N_VDestroy(cv_y);
    N_VDestroy(abst);
    SUNLinSolFree(LS);
    SUNMatDestroy(A);
  }

  auto copy_nvect(const N_Vector a, vector_t &b) {
    for (int i = 0; i < N; ++i) {
      b(i) = NV_Ith_S(a, i);
    }
  }

  auto initialize(vector_t &y0) {
    // set system size
    N = y0.rows();

    // allocate solution vector
    // y = N_VNew_Serial(N);

    // cv_y = N_VMake_Serial(N, static_cast<Eigen::VectorXd>(y0).data());
    cv_y = N_VMake_Serial(N, y0.data(), sunctx);
    // cv_y = N_VNew_Serial(N);

    // fill solution vector
    // copy_evect(y0, cv_y);

    // linear solver data
    // TODO: implement sparse
    A = SUNDenseMatrix(N, N, sunctx);
    LS = SUNLinSol_Dense(cv_y, A, sunctx);

    // init a new cvode context
    cvode_mem = CVodeCreate(cv_lmm[options.step_method], sunctx);
    // init integrator memory

    CVodeInit(
        cvode_mem,
        [](realtype t, N_Vector y, N_Vector ydot, void *user_data) {
          System *sys = static_cast<System *>(user_data);
          vector_map_t yref(NV_DATA_S(y), NV_LENGTH_S(y));
          vector_map_t fref(NV_DATA_S(ydot), NV_LENGTH_S(ydot));

          fref = sys->f(yref, t);

          return 0;
        },
        options.t0, cv_y);
    // set max step attempts
    CVodeSetMaxNumSteps(cvode_mem, options.max_steps);
    // set max/min step sizes
    CVodeSetMinStep(cvode_mem, options.step_min);
    CVodeSetMaxStep(cvode_mem, options.step_max);
    // set max order
    CVodeSetMaxOrd(cvode_mem, options.max_order);
    // set tolarances
    // TODO: allow vector tolerance
    abst = N_VNew_Serial(N, sunctx);
    NV_Ith_S(abst, 0) = 1.0E-8;
    NV_Ith_S(abst, 1) = 1.0E-14;
    NV_Ith_S(abst, 2) = 1.0E-6;
    CVodeSVtolerances(cvode_mem, options.rtol, abst);
    // CVodeSStolerances(cvode_mem, options.rtol, options.atol);
    //  set linear solver
    CVodeSetLinearSolver(cvode_mem, LS, A);
    // set jacobin
    CVodeSetJacFn(cvode_mem, [](realtype t, N_Vector y, N_Vector fy,
                                SUNMatrix J, void *user_data, N_Vector tmp1,
                                N_Vector tmp2, N_Vector tmp3) {
      System *sys = static_cast<System *>(user_data);

      vector_map_t yref(NV_DATA_S(y), NV_LENGTH_S(y));
      matrix_map_t Jref(SM_DATA_D(J), SM_ROWS_D(J), SM_COLUMNS_D(J));

      Jref = sys->J(yref, t);
      return 0;
    });
  }

  auto print_sol(double sol_time) {
    fmt::print("At t = {} y = {} {} {}\n", sol_time, NV_Ith_S(cv_y, 0),
               NV_Ith_S(cv_y, 1), NV_Ith_S(cv_y, 2));
  }

  auto step() {}

  auto check_retval(void *returnvalue, const char *funcname, int opt) {
    int *retval;

    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if (opt == 0 && returnvalue == NULL) {
      fmt::print(stderr,
                 "\nSUNDIALS_ERROR: {} failed - returned NULL pointer\n\n",
                 funcname);
      return (1);
    }

    /* Check if retval < 0 */
    else if (opt == 1) {
      retval = (int *)returnvalue;
      if (*retval < 0) {
        fmt::print(stderr, "\nSUNDIALS_ERROR: {} failed with retval = %d\n\n",
                   funcname, *retval);
        return (1);
      }
    }

    /* Check if function returned NULL pointer - no memory allocated */
    else if (opt == 2 && returnvalue == NULL) {
      fmt::print(stderr,
                 "\nMEMORY_ERROR: {} failed - returned NULL pointer\n\n",
                 funcname);
      return (1);
    }

    return (0);
  }
  auto letsgo() {
    auto iout = 0;
    auto tout = 0.4;
    double sol_time;
    while (1) {
      auto rstep = CVode(cvode_mem, tout, cv_y, &sol_time, CV_NORMAL);
      print_sol(sol_time);
      if (check_retval(&rstep, "CVode", 1))
        break;
      if (rstep == CV_SUCCESS) {
        iout++;
        tout *= 10.0;
      }
      if (iout == 12)
        break;
    }
    check_ans(cv_y, sol_time, options.rtol, abst);
  }
};

} // namespace cvode_wrapper
