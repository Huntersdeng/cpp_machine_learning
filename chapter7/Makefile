test: metric.o test.o
	g++ metric.o test.o -o test
	cp test ../test

metric.o : 
	g++ -c ../chapter2/metric.cpp

test.o :
	g++ -c test.cpp

clean: 
	rm -f metric.o test.o test ../test