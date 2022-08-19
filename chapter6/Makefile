test : kernel.o svm.o metric.o test.o
	g++ kernel.o svm.o metric.o test.o -o test
	cp test ../test

kernel.o: kernel.h kernel.cpp
	g++ -c kernel.cpp

svm.o : svm.h svm.cpp
	g++ -c svm.cpp

metric.o : ../chapter2/metric.h ../chapter2/metric.cpp
	g++ -c ../chapter2/metric.cpp

test.o : test.cpp
	g++ -c test.cpp

clean: 
	rm -f kernel.o svm.o metric.o test.o test ../test 