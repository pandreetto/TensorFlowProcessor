########################################################
# cmake file for building MuonCVXDDigitiser Package
CMAKE_MINIMUM_REQUIRED(VERSION 3.11 FATAL_ERROR)
########################################################

# project name
PROJECT(TensorFlowProcessor)


# project version
SET( ${PROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${PROJECT_NAME}_VERSION_MINOR 1 )
SET( ${PROJECT_NAME}_VERSION_PATCH 0 )

### BUILD OPTIONS ###########################################################

set(PACKAGE_INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
set(PACKAGE_INSTALL_INC_DIR include CACHE PATH "Installation directory for headers")

### DEPENDENCIES ############################################################

FIND_PACKAGE(ILCUTIL REQUIRED COMPONENTS ILCSOFT_CMAKE_MODULES)

# load default settings from ILCSOFT_CMAKE_MODULES
INCLUDE( ilcsoft_default_settings )

FIND_PACKAGE(Marlin 1.0 REQUIRED)
INCLUDE_DIRECTORIES(${Marlin_INCLUDE_DIRS})
LINK_LIBRARIES(${Marlin_LIBRARIES})
ADD_DEFINITIONS(${Marlin_DEFINITIONS})

FIND_PACKAGE(LCIO REQUIRED)
INCLUDE_DIRECTORIES(${LCIO_INCLUDE_DIRS})
LINK_LIBRARIES(${LCIO_LIBRARIES})
MESSAGE(STATUS "LCIO libs: ${LCIO_LIBRARIES}")

FIND_PACKAGE(tensorflow-lite 2.7.0 REQUIRED)
INCLUDE_DIRECTORIES(${tensorflow-lite_INCLUDE_DIRS})
LINK_LIBRARIES(${tensorflow-lite_LIBRARIES})
LINK_LIBRARIES("/usr/lib64/libruy.a"
               "/usr/lib64/libXNNPACK.a"
               "/usr/lib64/libflatbuffers.a"
               "/usr/lib64/libclog.a"
               "/usr/lib64/libcpuinfo.a"
               "/usr/lib64/libpthreadpool.a"
               "/usr/lib64/libfarmhash.a"
               "/usr/lib64/libfft2d_fftsg.a"
               "/usr/lib64/libfft2d_fftsg2d.a"
               "/usr/lib64/libdl.so")

#FIND_PACKAGE(Eigen3 3.3.7 REQUIRED)
#INCLUDE_DIRECTORIES(${Eigen3_INCLUDE_DIRS})
#LINK_LIBRARIES(${Eigen3_LIBRARIES})

FIND_PACKAGE(OpenMP)
option(OPENMP_ENABLED "Activate OpenMP parallel compilation" ON)
if(OPENMP_FOUND AND OPENMP_ENABLED)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
endif()

### LIBRARY #################################################################

# definitions to pass to the compiler
ADD_DEFINITIONS( "-Wall -ansi -pedantic -std=gnu++11" )
ADD_DEFINITIONS( "-Wno-long-long -Wno-effc++ -Wno-unused-parameter" )

# include directories
INCLUDE_DIRECTORIES( ./include )

# add library
ADD_SHARED_LIBRARY(${PROJECT_NAME} src/TensorFlowProcessor.cc)
INSTALL_SHARED_LIBRARY(${PROJECT_NAME} DESTINATION ${PACKAGE_INSTALL_LIB_DIR})

# display some variables and write them to cache
DISPLAY_STD_VARIABLES()

