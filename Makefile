CC:=g++
CXXFLAGS:=-std=c++11 -c -Wall -Isrc -O3

SOURCES:=$(wildcard src/*.cc)
OBJECTS:=$(patsubst %.cc, %.o, $(SOURCES))

all: $(OBJECTS)
	$(CC) $(OBJECTS) -lgmp -lgmpxx -lpthread

%.o : %.cc
	$(CC) -o $@ $(CXXFLAGS) $<
	
clean:
	rm -f $(OBJECTS)
	

