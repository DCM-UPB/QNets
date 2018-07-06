#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([DCM-FFNN], [0.3], [ithanil@mail.uni-paderborn.de])
: ${CXXFLAGS=""}
#AC_CONFIG_SRCDIR([include/InputUnit.hpp])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIRS([m4])
LT_INIT

# Checks for programs.
AC_PROG_CXX
AC_PROG_CPP
AC_PROG_LN_S

# Checks for libraries.

# Checks for header files.

AC_CHECK_HEADERS([gsl/gsl_vector.h gsl/gsl_matrix.h gsl/gsl_blas.h gsl/gsl_multifit_nlinear.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_SIZE_T

# Checks for library functions.
AC_CHECK_FUNCS([pow sqrt])

# Add debug support
AC_ARG_ENABLE(debug,
  AS_HELP_STRING(
    [--enable-debug],
    [enable debugging, default: no]),
    [case "${enableval}" in
      yes) debug=true ;;
      no)  debug=false ;;
      *)   AC_MSG_ERROR([bad value ${enableval} for --enable-debug]) ;;
    esac],
    [debug=false])
AM_CONDITIONAL(DEBUG, test x"$debug" = x"true")
AM_COND_IF(DEBUG,
    AC_DEFINE(DEBUG, 1, [Define to 0 if this is a release build]),
    AC_DEFINE(DEBUG, 0, [Define to 1 or higher if this is a debug build]))


# Add debug support
AC_ARG_ENABLE(coverage,
  AS_HELP_STRING(
    [--enable-coverage],
    [enable coverage, default: no]),
    [case "${enableval}" in
      yes) coverage=true ;;
      no)  coverage=false ;;
      *)   AC_MSG_ERROR([bad value ${enableval} for --enable-coverage]) ;;
    esac],
    [debug=false])
AM_CONDITIONAL(COVERAGE, test x"$coverage" = x"true")
AM_COND_IF(COVERAGE,
    AC_DEFINE(COVERAGE, 1, [Define to 0 if you don't need code coverage statistics.]),
    AC_DEFINE(COVERAGE, 0, [Define to 1 or higher if you want to generate code coverage statistics.]))

AC_CONFIG_FILES([Makefile
                 benchmark/Makefile
                 benchmark/bench_actfs_derivs/Makefile
                 benchmark/bench_actfs_ffprop/Makefile
                 bin/Makefile
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
