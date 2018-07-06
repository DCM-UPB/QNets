if DEBUG
  AM_CXXFLAGS += $(DEBUGFLAGS)
else
  AM_CXXFLAGS += $(OPTFLAGS)
endif

if COVERAGE
  AM_CPPFLAGS += $(COVCFLAGS)
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
