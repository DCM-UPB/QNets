SUBDIRS = bench_actfs_derivs bench_actfs_ffprop

run-benchmarks:
	for dir in $(SUBDIRS); do \
	  if [[ "$$dir" == *"bench_"* ]]; then $(MAKE) -C $$dir run-benchmark; fi; \
	done
