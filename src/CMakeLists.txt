file(GLOB SOURCES "*.cpp" "*/*.cpp")
add_library(qnets SHARED ${SOURCES})
target_link_libraries(qnets "${GSL_LIBRARIES}" "${OpenMP_CXX_LIBRARIES}") # shared libs
add_library(qnets_static STATIC ${SOURCES})
target_link_libraries(qnets_static "${GSL_LIBRARIES}" "${OpenMP_CXX_LIBRARIES}") # static (+ some shared) libs
