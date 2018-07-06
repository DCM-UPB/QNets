#if !DEBUG
#  if !COVERAGE
#  AM_CXXFLAGS += $(OPTFLAGS)
#  endif
#endif

if COVERAGE
  include $(top_srcdir)/aminclude_static.am
  AM_CPPFLAGS += $(CODE_COVERAGE_CPPFLAGS)
  AM_CXXFLAGS += $(CODE_COVERAGE_CXXFLAGS)
endif

if PROFILING
  AM_CPPFLAGS += $(PROFFLAGS)
endif

if OPENMP
  AM_CPPFLAGS += $(OMPCFLAGS)
endif

noinst_LTLIBRARIES = libffnn.la
libffnn_ladir = ../
include headers.am
include sources.am
