include test.am

SUBDIRS = ut1 ut2 ut3 ut4 ut5 ut6 ut7 ut8 ut9

TESTS = exe ut1/exe ut2/exe ut3/exe ut4/exe ut5/exe ut6/exe ut7/exe ut8/exe ut9/exe
AUTOMAKE_OPTIONS = parallel-tests
if !OPENMP
if VALGRIND_ENABLED
  @VALGRIND_CHECK_RULES@
  LOG_COMPILER = libtool e $(VALGRIND) --leak-check=full --track-origins=yes
else
  LOG_COMPILER = libtool e
endif
endif
