test: dtree.o metric.o test.o
	g++ dtree.o metric.o test.o -o test
	cp test ../test

metric.o : 
	g++ -c ../chapter2/metric.cpp

dtree.o: 
	g++ -c ./dtree.cpp

test.o :
	g++ -c test.cpp

clean: 
	rm -f dtree.o metric.o test.o test ../test