#include <fmt/core.h>
#include <string_view>
#include <chrono>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>

#include "types.hh"
#include "cvode_wrapper.hh"

using namespace types;

struct System
{
    template<class T>
    inline auto f(Eigen::MatrixBase<T>& y, double t)
    {
        using mt = T::PlainObject;
        mt ydot(y.rows());

        ydot(0) = -0.04*y(0) + 1.0E4*y(1)*y(2);
        ydot(2) = 3.0E7*y(1)*y(1);
        ydot(1) = -ydot(0) - ydot(2);

    }

    template<class T>
    inline auto J(Eigen::MatrixBase<T>& y, double t)
    {
        auto _f = [this](auto&& ...args){ return f(std::forward<decltype(args)>(args)...); };
        dvector_t dydt(y.size());
        dvector_t yy = y;
        matrix_t J = autodiff::jacobian(_f, autodiff::wrt(yy), autodiff::at(yy,t), dydt);
        return J;
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