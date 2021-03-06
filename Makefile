RM=rm
CC=nvcc
CFLAGS=-c
LDFLAGS=-lGL -lGLU -lglut 
PROJECTNAME=GameOfLife

all: $(PROJECTNAME)

GameOfLife: main.o GameLogic.o
	$(CC) -o $(PROJECTNAME) main.o GameLogic.o $(LDFLAGS)

main.o: main.cu
	$(CC) $(CFLAGS) main.cu -o main.o

GameLogic.o: GameLogic.cu
	$(CC) $(CFLAGS) GameLogic.cu -o GameLogic.o

clean:
	$(RM) -r *.o 
	$(RM) $(PROJECTNAME)

run: all
	./$(PROJECTNAME)
