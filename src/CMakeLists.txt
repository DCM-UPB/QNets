file(GLOB SOURCES "*/*.cpp")
add_library(ffnn SHARED ${SOURCES})
target_link_libraries(ffnn "${GSL_LIBRARIES}" "${OpenMP_CXX_LIBRARIES}") # shared libs
add_library(ffnn_static STATIC ${SOURCES})
target_link_libraries(ffnn_static "${GSL_LIBRARIES}" "${OpenMP_CXX_LIBRARIES}") # static (+ some shared) libs
