.PHONY: clean

ALL_CPP = $(wildcard */*.cpp)
SRC_CPP = $(filter-out $(wildcard demo/*.cpp), $(ALL_CPP))
OBJ_O = $(patsubst %.cpp,%.o, $(SRC_CPP))
OBJ_H =  $(wildcard *.h */*.h) 
OBJ_TARGET = $(patsubst %.cpp,%.o,$(wildcard demo/*.cpp))
EXEC_TMP = $(patsubst %.cpp,%,$(wildcard demo/*.cpp))
EXEC_OUT := $(subst /,_,$(EXEC_TMP) )

all: $(OBJ_O) $(OBJ_H)

chapter%: $(OBJ_O) ./demo/chapter%.o
	g++ $^ -o demo_$@

%.o : %.cpp
	g++ -c $^ -o $@

clean:
	rm -f $(OBJ_O) $(OBJ_TARGET) $(EXEC_OUT)
