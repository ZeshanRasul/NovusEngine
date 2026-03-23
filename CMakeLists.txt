cmake_minimum_required(VERSION 3.10)

project(NovusEngine)

find_package(fmt CONFIG REQUIRED)

add_executable(NovusEngine main.cpp)

target_link_libraries(NovusEngine PRIVATE fmt::fmt)