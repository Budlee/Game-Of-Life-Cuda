RM=rm
CC=nvcc
CFLAGS=-c
LDFLAGS=-lGL -lGLU -lglut -lGLEW
PROJECTNAME=GameOfLife

all: $(PROJECTNAME)

GameOfLife: main.o GameLogic.o
	$(CC) -o $(PROJECTNAME) main.o GameLogic.o $(LDFLAGS)

main.o: main.c
	$(CC) $(CFLAGS) main.c -g -o main.o

GameLogic.o: GameLogic.c
	$(CC) $(CFLAGS) GameLogic.c -g -o GameLogic.o

clean:
	$(RM) -r *.o 
	$(RM) $(PROJECTNAME)

run: all
	./$(PROJECTNAME)
