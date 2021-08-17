project(cvode_eigen)
cmake_minimum_required(VERSION 3.18)

enable_language(CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

find_package(fmt REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
find_package(SUNDIALS 5.7 REQUIRED)

set (debug "$<CONFIG:Debug>")
set (release "$<CONFIG:Release>")

add_library(cve::flags INTERFACE IMPORTED)

target_compile_features(cve::flags
    INTERFACE
        cxx_std_20
)

target_compile_options(cve::flags
    INTERFACE
        "-fmax-errors=2"
        $<${release}:
            "-Ofast"
        >
)

target_include_directories(cve::flags
INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/autodiff>
)
target_link_libraries(cve::flags
    INTERFACE
        SUNDIALS::CVODE
        SUNDIALS::NVECS
        Eigen3::Eigen
        fmt::fmt
)

set (EXE_LBL "cve.x")
add_executable(${EXE_LBL})

target_sources(${EXE_LBL}
    PRIVATE
        src/main.cc
)

target_link_libraries(${EXE_LBL}
    PRIVATE
        cve::flags
)
