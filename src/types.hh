#pragma once

#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>

namespace types{

using dvector_t = autodiff::VectorXreal;
using vector_t = Eigen::VectorXd;
using matrix_t = Eigen::MatrixXd;


using vector_map_t = Eigen::Map<vector_t>;
using matrix_map_t = Eigen::Map<matrix_t>;

} // namespace types