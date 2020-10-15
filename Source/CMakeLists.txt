cmake_minimum_required(VERSION 2.8.9)
project(StackedAlternatingAMQs)
 
#Bring the headers, such as Student.h into the project
include_directories(
        ${PROJECT_SOURCE_DIR}/Source
        ${PROJECT_SOURCE_DIR}/Headers
)

#However, the file(GLOB...) allows for wildcard additions:
file(GLOB all_SRCS
        "${PROJECT_SOURCE_DIR}/Headers/*.h"
        "${PROJECT_SOURCE_DIR}/Source/*.cpp"
        
 
add_executable(DataGenerator ${all_SRCS} ${PROJECT_SOURCE_DIR}/Source/DataGenerator.cpp ${PROJECT_SOURCE_DIR}/Source/StackedBloomFilter.h)
