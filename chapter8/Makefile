.PHONY: clean

OBJ = $(patsubst %.cpp,%.o,$(wildcard *.cpp ../chapter2/*.cpp))
target = test

$(target): $(OBJ)
	g++ $(OBJ) -o $(target)
	cp $(target) ../$(target)

%.o : %.cpp
	g++ -c $^ -o $@

clean: 
	rm -f $(OBJ) $(target) ../$(target)