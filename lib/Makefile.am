if !DEBUG
if !COVERAGE
  AM_CXXFLAGS += $(OPTFLAGS)
endif
endif

include $(top_srcdir)/aminclude_static.am

if COVERAGE
  AM_CPPFLAGS += $(COVERAGE_CPPFLAGS)
  AM_CXXFLAGS += $(COVERAGE_CXXFLAGS)
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


source-lists:
	echo "libffnn_la_HEADERS = \\" > headers.am
	find ../src/ -name *.hpp | tr '\n' ' ' >> headers.am
	echo "libffnn_la_SOURCES = \\" > sources.am
	find ../src/ -name *.cpp | tr '\n' ' ' >> sources.am

.PHONY: source-lists
