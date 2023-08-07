cmake_minimum_required(VERSION 3.18)
project(cvode_eigen)

enable_language(CXX)
include(CMakeDetermineCXXCompiler)
include(CheckCXXCompilerFlag)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

find_package(fmt REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
find_package(SUNDIALS 5.7 REQUIRED)
find_package(autodiff)

set(debug "$<CONFIG:Debug>")
set(release "$<CONFIG:Release>")

set(EXE_LBL "cve.x")
add_executable(${EXE_LBL})

target_sources(${EXE_LBL} PRIVATE src/main.cc)

target_compile_features(${EXE_LBL} PRIVATE cxx_std_20)

target_compile_options(
  ${EXE_LBL}
  PRIVATE "-fmax-errors=2"
          $<${release}:
          "-Ofast"
          >
          $<${debug}:
          "-g"
          >)

# target_include_directories(cve::flags INTERFACE
# $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/autodiff> )
target_link_libraries(
  ${EXE_LBL} PRIVATE SUNDIALS::CVODE SUNDIALS::NVECS Eigen3::Eigen
                     autodiff::autodiff fmt::fmt)
