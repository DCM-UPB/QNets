include_directories(common/)
link_libraries(qnets)

add_executable(bench_actfs_derivs bench_actfs_derivs/main.cpp)
add_executable(bench_actfs_ffprop bench_actfs_ffprop/main.cpp)
add_executable(bench_nunits_ffprop bench_nunits_ffprop/main.cpp)
add_executable(bench_templ_ffprop bench_templ_ffprop/main.cpp)
