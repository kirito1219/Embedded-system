CC = g++
CFLAGS = `pkg-config --cflags opencv4`
LIBS = `pkg-config --libs opencv4`

detect:detect.cpp
	$(CC) -std=c++11 detect.cpp -o detect $(LIBS) $(CFLAGS)

