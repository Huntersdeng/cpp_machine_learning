.PHONY: clean

OBJ_O = $(patsubst %.cpp,%.o,$(wildcard ../chapter2/*.cpp ../chapter3/*.cpp))
OBJ_H =  $(wildcard ../chapter2/*.h ../chapter3/*.h)
OBJ_TARGET = $(patsubst %.cpp,%.o,$(wildcard *.cpp))
EXEC_TARGET = $(patsubst %.cpp,%,$(wildcard *.cpp))

chapter3: $(OBJ_O) $(OBJ_H) chapter3.o
	g++ $^ -o $@
%.o : %.cpp
	g++ -c $^ -o $@

clean:
	rm -f $(OBJ_O) $(OBJ_TARGET) $(EXEC_TARGET)