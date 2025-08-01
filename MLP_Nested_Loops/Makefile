CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra
TARGET = main
SOURCES = NeuralNetwork.cpp Neuron.cpp Layer.cpp main.cpp
OBJECTS = $(SOURCES:.cpp=.o)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: clean
