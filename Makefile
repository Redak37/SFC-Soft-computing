CC = gcc
CFLAGS = -std=c17 -pedantic -Wall -Wextra -O2

CXX = g++
CXXFLAGS = -std=c++17 -pedantic -Wall -Wextra -O2

all: rbf

run: run1

run1:
	./rbf -n 4

run2:
	./rbf -r -n 4

run3:
	./rbf -r

run4:
	./rbf -f input2 -r -n 8 -m 0.25

clean:
	rm rbf xducho07.zip

zip:
	zip xducho07.zip doc.pdf Makefile rbf.cpp rbf.hpp input input2
