all: scale

scale: scale.o
	g++ scale.o -o scale -lpthread -lboost_system

scale.o: scale.cpp
	g++ -c -Os scale.cpp -o scale.o 

clean:
	rm scale *.o
