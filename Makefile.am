ACLOCAL_AMFLAGS = -I m4

SUBDIRS = include lib test benchmark examples

benchmark:
	cd benchmark && make benchmark

include-links:
	cd include && make include-links

source-lists:
	cd lib && make source-lists

update-sources: include-links source-lists

.PHONY: benchmark source-lists include-links update-sources
