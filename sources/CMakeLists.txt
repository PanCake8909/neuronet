#output filename
set(PROJECT_NAME nerualnet.rubick)


#project files without ","
set(SOURCES  main.cpp  net.cpp  net.h  neuron.cpp  neuron.h  trainingSet.cpp  trainingSet.h)

#option libs without ","
set(LIBRARIES your_libraries_here)


#flags without ","
#processor instruction flags: -msse, -msse2, -msse3, -mmmx, -m3dnow -march='your CPU here'
#default is auto-definition of march: -march=native

#warning: if you use flag '-march', binary file can not work in else CPU!!!
#warning: don't use '-march-native' flag with distcc compiller!!!

#you should use   $cat /proc/cpuinfo | tr " " "\012" |grep -e sse -e mmx -e now -e pni | sort -u
#to be confident that your processor can use instructions '-msse, -msse2, -msse3, -mmmx, -m3dnow'
#just switch on CPU flags which 'fit' for you CPU
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -w -pipe -std=c++11 -std=c++14 -std=c++17 -march=native -mmmx -msse -msse2 -msse3 -msse4")




project(${PROJECT_NAME})
add_executable(${PROJECT_NAME} ${SOURCES})

#uncomment it if you need to link with option libs
#target_link_libraries(${PROJECT_NAME} ${LIBRARIES})
