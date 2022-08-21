test: dtree.o metric.o test.o
	g++ dtree.o metric.o test.o -o test
	cp test ../test

metric.o : ../chapter2/metric.cpp
	g++ -c ../chapter2/metric.cpp

dtree.o: dtree.cpp
	g++ -c dtree.cpp

test.o : test.cpp
	g++ -c test.cpp

clean: 
	rm -f dtree.o metric.o test.o test ../test