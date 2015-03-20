RM=rm
CC=nvcc
CFLAGS=-c
LDFLAGS=-lGL -lGLU -lglut -lGLEW
PROJECTNAME=GameOfLife

all: $(PROJECTNAME)

GameOfLife: main.o GameLogic.o
	$(CC) -o $(PROJECTNAME) main.o GameLogic.o $(LDFLAGS)

main.o: main.cu
	$(CC) $(CFLAGS) main.cu -g -o main.o

GameLogic.o: GameLogic.cu
	$(CC) $(CFLAGS) GameLogic.cu -g -o GameLogic.o

clean:
	$(RM) -r *.o 
	$(RM) $(PROJECTNAME)

run: all
	./$(PROJECTNAME)
