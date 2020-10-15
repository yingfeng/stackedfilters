cmake_minimum_required(VERSION 2.8.9)
project(StackedAlternatingAMQs CXX)

set(CMAKE_CXX_FLAGS " -O3 -Wall -Wextra")

#Bring the headers, such as Student.h into the project
include_directories(
        ${PROJECT_SOURCE_DIR}/Source
        ${PROJECT_SOURCE_DIR}/Headers
        ${PROJECT_SOURCE_DIR}/cqf-master/include
        ${PROJECT_SOURCE_DIR}/cqf-master/src
        ${PROJECT_SOURCE_DIR}/stacked-filters-optimization-clean/include
)

#However, the file(GLOB...) allows for wildcard additions:
file(GLOB all_SRCS
        "${PROJECT_SOURCE_DIR}/Headers/*.h"
        "${PROJECT_SOURCE_DIR}/Source/*.cpp"
        "${PROJECT_SOURCE_DIR}/Headers/*.h"
        "${PROJECT_SOURCE_DIR}/Source/*.c")


add_executable(DataGenerator "${PROJECT_SOURCE_DIR}/Main/DataGenerator.cpp" ${all_SRCS})
target_link_libraries(DataGenerator ${PROJECT_SOURCE_DIR}/Libs/libnlopt.a)

add_executable(DataSizeGenerator "${PROJECT_SOURCE_DIR}/Main/DataSizeGenerator.cpp" ${all_SRCS})
target_link_libraries(DataSizeGenerator ${PROJECT_SOURCE_DIR}/Libs/libnlopt.a)

add_executable(ZipfGraphsDataGenerator "${PROJECT_SOURCE_DIR}/Main/ZipfGraphsDataGenerator.cpp" ${all_SRCS})
target_link_libraries(ZipfGraphsDataGenerator ${PROJECT_SOURCE_DIR}/Libs/libnlopt.a)

add_executable(ZipfGraphsDataGeneratorDiskVersion "${PROJECT_SOURCE_DIR}/Main/ZipfGraphsDataGeneratorDiskVersion.cpp" ${all_SRCS})
target_link_libraries(ZipfGraphsDataGeneratorDiskVersion ${PROJECT_SOURCE_DIR}/Libs/libnlopt.a)

add_executable(FirehoseDataGenerator "${PROJECT_SOURCE_DIR}/Main/FirehoseDataGenerator.cpp" ${all_SRCS})
target_link_libraries(FirehoseDataGenerator ${PROJECT_SOURCE_DIR}/Libs/libnlopt.a)

add_executable(FirehoseDataGeneratorDiskVersion "${PROJECT_SOURCE_DIR}/Main/FirehoseDataGeneratorDiskVersion.cpp" ${all_SRCS})
target_link_libraries(FirehoseDataGeneratorDiskVersion ${PROJECT_SOURCE_DIR}/Libs/libnlopt.a)

add_executable(FirehoseDataGeneratorDiskVersion_2 "${PROJECT_SOURCE_DIR}/Main/FirehoseDataGeneratorDiskVersion2.cpp" ${all_SRCS})
target_link_libraries(FirehoseDataGeneratorDiskVersion_2 ${PROJECT_SOURCE_DIR}/Libs/libnlopt.a)

add_executable(URLBlacklistingDiskVersion "${PROJECT_SOURCE_DIR}/Main/URULBlacklistingDiskVersion.cpp" ${all_SRCS})
target_link_libraries(URLBlacklistingDiskVersion ${PROJECT_SOURCE_DIR}/Libs/libnlopt.a)

add_executable(ZipfDiskVersionAdaptive "${PROJECT_SOURCE_DIR}/Main/ZipfDiskVersionAdaptive.cpp" ${all_SRCS})
target_link_libraries(ZipfDiskVersionAdaptive ${PROJECT_SOURCE_DIR}/Libs/libnlopt.a)

add_executable(ZipfDiskVersionAdaptiveStages "${PROJECT_SOURCE_DIR}/Main/ZipfDiskVersionAdaptiveStages.cpp" ${all_SRCS})
target_link_libraries(ZipfDiskVersionAdaptiveStages ${PROJECT_SOURCE_DIR}/Libs/libnlopt.a)

add_executable(ZipfGraphsDataGeneratorMemoryVersion "${PROJECT_SOURCE_DIR}/Main/ZipfGraphsDataGeneratorMemoryVersion.cpp" ${all_SRCS})
target_link_libraries(ZipfGraphsDataGeneratorMemoryVersion ${PROJECT_SOURCE_DIR}/Libs/libnlopt.a)

add_executable(CQFDataGenerator "${PROJECT_SOURCE_DIR}/Main/CQFDataGenerator.cpp" ${all_SRCS})
target_link_libraries(CQFDataGenerator ${PROJECT_SOURCE_DIR}/Libs/libnlopt.a)

add_executable(CuckooDataGenerator "${PROJECT_SOURCE_DIR}/Main/CuckooDataGenerator.cpp" ${all_SRCS})
target_link_libraries(CuckooDataGenerator ${PROJECT_SOURCE_DIR}/Libs/libnlopt.a)

add_executable(InsertsDataGenerator "${PROJECT_SOURCE_DIR}/Main/InsertsDataGenerator.cpp" ${all_SRCS})
target_link_libraries(InsertsDataGenerator ${PROJECT_SOURCE_DIR}/Libs/libnlopt.a)

add_executable(InsertsDataGenerator2 "${PROJECT_SOURCE_DIR}/Main/InsertsDataGenerator2.cpp" ${all_SRCS})
target_link_libraries(InsertsDataGenerator2 ${PROJECT_SOURCE_DIR}/Libs/libnlopt.a)

add_executable(CQFTest "${PROJECT_SOURCE_DIR}/Main/CQFTest.cpp" ${all_SRCS})
target_link_libraries(CQFTest ${PROJECT_SOURCE_DIR}/Libs/libnlopt.a)

add_executable(URLBlacklisting "${PROJECT_SOURCE_DIR}/Main/URLBlacklisting.cpp" ${all_SRCS})
target_link_libraries(URLBlacklisting ${PROJECT_SOURCE_DIR}/Libs/libnlopt.a)

add_executable(RetouchedExperimentation "${PROJECT_SOURCE_DIR}/Main/RetouchedExperimentation.cpp" ${all_SRCS})
target_link_libraries(RetouchedExperimentation ${PROJECT_SOURCE_DIR}/Libs/libnlopt.a)

add_executable(UniformDataGenerator "${PROJECT_SOURCE_DIR}/Main/UniformDataGenerator.cpp" ${all_SRCS})
target_link_libraries(UniformDataGenerator ${PROJECT_SOURCE_DIR}/Libs/libnlopt.a)

add_executable(ASFOptimizationTest "${PROJECT_SOURCE_DIR}/Main/ASFOptimizationTest.cpp" ${all_SRCS})
target_link_libraries(ASFOptimizationTest ${PROJECT_SOURCE_DIR}/Libs/libnlopt.a)