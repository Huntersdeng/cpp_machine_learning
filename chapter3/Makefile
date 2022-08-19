test: metric.o test.o
	g++ metric.o test.o -o test
	cp test ../test

metric.o : ../chapter2/metric.h ../chapter2/metric.cpp
	g++ -c ../chapter2/metric.cpp

test.o : test.cpp
	g++ -c test.cpp

clean: 
	rm -f metric.o test.o test ../test