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
  AM_CXXFLAGS += $(PROFFLAGS)
endif

if OPENMP
  AM_CPPFLAGS += $(OMPCFLAGS)
endif

lib_LTLIBRARIES = libffnn.la
include headers.am
include sources.am


source-lists:
	echo "pkginclude_HEADERS = \\" > headers.am
	find ../src/ -name *.hpp | tr '\n' ' ' >> headers.am
	echo "libffnn_la_SOURCES = \\" > sources.am
	find ../src/ -name *.cpp | tr '\n' ' ' >> sources.am

.PHONY: source-lists
