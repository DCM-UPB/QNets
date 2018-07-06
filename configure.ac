#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([DCM-FFNN], [0.3], [ithanil@mail.uni-paderborn.de])
: ${CXXFLAGS=""} # delete default optimization user flags
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIRS([m4])
AC_LANG([C++])


# --- Options
#m4_pattern_allow([AC_MSG_ERROR])

# Add debug support
m4_include([m4/ax_check_enable_debug.m4])
AX_CHECK_ENABLE_DEBUG()
AM_CONDITIONAL(DEBUG, test x"$ax_enable_debug" = x"yes")
AM_COND_IF(DEBUG,
    AC_DEFINE(DEBUG, 1, [Define to 0 if this is a release build]),
    AC_DEFINE(DEBUG, 0, [Define to 1 or higher if this is a debug build]))


# Add coverage support (gcov)
#m4_include([m4/ax_code_coverage.m4])
#AX_CODE_COVERAGE()
AM_CONDITIONAL(COVERAGE, test x"$enable_code_coverage" = x"true")
AM_COND_IF(COVERAGE,
    AC_DEFINE(COVERAGE, 1, [Define to 0 if you don't need code coverage statistics.]),
    AC_DEFINE(COVERAGE, 0, [Define to 1 or higher if you want to generate code coverage statistics.]))


# Add profiling support (gperftools)
AC_ARG_ENABLE(profiling,
  AS_HELP_STRING(
    [--enable-profiling],
    [enable profiling, default: no]),
    [case "${enableval}" in
      yes) profiling=true ;;
      no)  profiling=false ;;
    esac],
    [profiling=false])
AM_CONDITIONAL(PROFILING, test x"$profiling" = x"true")
AM_COND_IF(PROFILING,
    AC_DEFINE(PROFILING, 1, [Define to 0 if you don't need performance profiling.]),
    AC_DEFINE(PROFILING, 0, [Define to 1 or higher if you want to generate performance profiles.]))


# Add OpenMP support
AC_ARG_ENABLE(openmp,
  AS_HELP_STRING(
    [--enable-openmp],
    [enable openmp, default: no]),
    [case "${enableval}" in
      yes) openmp=true ;;
      no)  openmp=false ;;
    esac],
    [openmp=false])
AM_CONDITIONAL(OPENMP, test x"$openmp" = x"true")
AM_COND_IF(OPENMP,
    AC_DEFINE(OPENMP, 1, [Define to 0 if you don't want to use threading via OpenMP.]),
    AC_DEFINE(OPENMP, 0, [Define to 1 or higher if you want to enable OpenMP threading.]))



# --- Checks

LT_INIT # use libtool

# Checks for programs.
AC_PROG_CXX
AC_PROG_CPP
AC_PROG_LN_S

# check for c++11 support
m4_include([m4/ax_cxx_compile_stdcxx.m4])
AX_CXX_COMPILE_STDCXX(11, noext, mandatory)

# Checks for libraries.

# Checks for header files.

AC_CHECK_HEADERS([gsl/gsl_vector.h gsl/gsl_matrix.h gsl/gsl_blas.h gsl/gsl_multifit_nlinear.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_SIZE_T

# Checks for library functions.
AC_CHECK_FUNCS([pow sqrt])



# --- Compiler flags

AC_SUBST([DEBUGFLAGS], ["-g -O0"])
AC_SUBST([OPTFLAGS], ["-O3 -flto"])
AC_SUBST([PROFFLAGS], ["-DWITHGPERFTOOLS -lprofiler"])
AC_SUBST([OMPCFLAGS], ["-DOPENMP -fopenmp"])

AC_SUBST([AM_CXXFLAGS], ["-std=c++11 -fpic -Wall"])
AC_SUBST([AM_CPPFLAGS], ["-I$(pwd)/include"])


# --- LDFLAGS

#AC_SUBST([COVLFLAGS], ["$(CODE_COVERAGE_LIBS)"])
AC_SUBST([OMPLFLAGS], ["-lomp"])

flags="-L$(pwd)/lib -lffnn -lgsl -lgslcblas"
AM_COND_IF(COVERAGE,
  flags+=" ${COVLFLAGS}")
AM_COND_IF(OPENMP,
  flags+=" ${OMPLFLAGS}")

AC_SUBST([AM_LDFLAGS], [$flags])



# --- Makefiles

AC_CONFIG_FILES([Makefile
                 benchmark/Makefile
                 benchmark/bench_actfs_derivs/Makefile
                 benchmark/bench_actfs_ffprop/Makefile
                 lib/Makefile
                 doc/Makefile
                 examples/Makefile
                 examples/ex1/Makefile
                 examples/ex2/Makefile
                 examples/ex3/Makefile
                 examples/ex4/Makefile
                 examples/ex5/Makefile
                 examples/ex6/Makefile
                 examples/ex7/Makefile
                 examples/ex8/Makefile
                 examples/ex9/Makefile
                 test/Makefile
                 test/ut1/Makefile
                 test/ut2/Makefile
                 test/ut3/Makefile
                 test/ut4/Makefile
                 test/ut5/Makefile
                 test/ut6/Makefile
                 test/ut7/Makefile
                 test/ut8/Makefile
                 test/ut9/Makefile])
AM_INIT_AUTOMAKE([subdir-objects])
AC_OUTPUT
