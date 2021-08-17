#include <fmt/core.h>
#include <string_view>
#include <chrono>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "cvode_wrapper.hh"

using vector_t = Eigen::VectorXd;
using matrix_t = Eigen::MatrixXd;

struct System
{
    template<class U>
    inline auto f(Eigen::MatrixBase<U>& y, double t)
    {
        vector_t ydot(y.rows());

        ydot(0) = -0.04*y(0) + 1.0E4*y(1)*y(2);
        ydot(2) = 3.0E7*y(1)*y(1);
        ydot(1) = -ydot(0) - ydot(2);

        return ydot;
    }

    template<class U>
    inline auto J(Eigen::MatrixBase<U>& y, double t)
    {
        matrix_t jac(y.rows(), y.rows());

        jac(0,0) = -0.04;
        jac(0,1) = 1.0E4 * y(2);
        jac(0,2) = 1.0E4 * y(1);

        jac(1,0) = 0.04;
        jac(1,1) = -1.0E4*y(2) - 6.0E7*y(1);
        jac(1,2) = -1.0E4*y(1);

        jac(2,0) = 0.0;
        jac(2,1) = 6.0E7*y(1);
        jac(2,2) = 0.0;

        return jac;
    }
};

int main()
{

    vector_t y0(3);
    y0(0) = 1.0; y0(1) = 0.0; y0(2) = 0.0;

    auto stepper = cvode_wrapper::cvode_stepper<System>(cvode_wrapper::cv_options{});
    stepper.initialize(y0);

    stepper.letsgo();
    return 0;
}