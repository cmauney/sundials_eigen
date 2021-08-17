#pragma once

#include <fmt/core.h>
#include <string_view>
#include <vector>
#include <concepts>


template <class S, class V>
concept ODESystem = requires(S s, V v)
{
    s.f(v);
};

struct ode__
{
    void f(double t)
    {
        return;
    };
};

struct stepper
{
    template<ODESystem<double> Sys>
    void step(Sys s)
    {
        s.f(1.0);
        return;
    }
};

ode__ sys;
stepper stp;

void tt()
{
    stp.step(sys);
}