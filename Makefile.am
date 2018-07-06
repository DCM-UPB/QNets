AUTOMAKE_OPTIONS = foreign
ACLOCAL_AMFLAGS = -I m4

SUBDIRS = lib test benchmark examples
#SUBDIRS = lib test benchmark examples doc

benchmark:
	cd benchmark && make benchmark && cd ..
.PHONY: benchmark
