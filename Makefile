#########################################################
#              quicksort Makefile             #
#########################################################
# edited by Ethan Nanavati
CXX      = g++
CXXFLAGS = -g3 -I "C:\ProgramData\chocolatey\lib\eigen\include\eigen3" -fopenmp #-Wall -Wextra -Wpedantic -Wshadow
LDFLAGS  = -g3 -I "C:\ProgramData\chocolatey\lib\eigen\include\eigen3" -fopenmp


# Metrosim rule 
randomizedQR: main.o randomized_QR.o
	${CXX} ${LDFLAGS} -o randomizedQR main.o randomized_QR.o

# This rule builds main.o	
main.o: main.cpp randomized_QR.h
	$(CXX) $(CXXFLAGS) -c main.cpp
	
randomized_QR.o: randomized_QR.cpp randomized_QR.h
	$(CXX) $(CXXFLAGS) -c randomized_QR.cpp

# 	# # The below rule will be used by unit_test.
#	# unit_test: unit_test_driver.o PassengerQueue.o Passenger.o MetroSim.o
#		# $(CXX) $(CXXFLAGS) $^
#	
#	
#	# # This rule provides your submission 
#	# provide:
#	# 	provide comp15 hw2_MetroSim PassengerQueue.h \
	# 				    PassengerQueue.cpp \
	# 				    Passenger.h Passenger.cpp  \
	# 				    unit_tests.h Makefile README \
	# 				    MetroSim.h MetroSim.cpp main.cpp

# remove executables, object code, and temporary files from the current folder 
# -- the executable created by unit_test is called a.out
clean: 
	rm *.o *~ a.out
