/*
 * File:   main.c
 * Author: matt
 *
 * Created on 16 March 2015, 10:42
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>

#include <GL/glew.h>
#include <GL/glu.h>
#include <GL/glut.h>



#include "GameLogic.h"

#define MOVE_DISTANCE 5
#define INTERVAL (1000 / 60)

// Works best as a square grid
uint32_t Y_AXIS = 200;
uint32_t X_AXIS = Y_AXIS;

uint8_t timmingVisible = 1;
float posX = 0.01, posY = -0.1, scale = 1.0;

void drawLines (void) {
    glClear(GL_COLOR_BUFFER_BIT);
    glLoadIdentity();
    glTranslatef(posX,posY,0.0);
    glScalef(scale, scale, 1.0);
    uint64_t yIndex = Y_AXIS;
    uint64_t xIndex = 0;
    uint8_t *cellsLocal = getGameOfLifeState();
    while (yIndex != 0)
    {
    	uint64_t localYIndex = yIndex -1;
        xIndex = 0;
        while (xIndex < X_AXIS)
        {
            if (cellsLocal[(localYIndex * X_AXIS) + xIndex] != 0)
            {
                glColor3f(0.0, 1.0, 0.0);
            }
            else
            {
                glColor3f(1.0, 1.0, 1.0);
            }
            glRecti(xIndex, localYIndex, xIndex+1, localYIndex+1);
            ++xIndex;
        }

        --yIndex;
    }
    glutSwapBuffers();
}

double get_time()
{
    struct timeval t;
    struct timezone tzp;
    gettimeofday(&t, &tzp);
    return t.tv_sec + t.tv_usec*1e-6;
}

void update(int value) {
   // Call update() again in 'interval' milliseconds
   glutTimerFunc(INTERVAL, update, 0);
   if(timmingVisible == 0)
   {
	   double start ,end;
	   start = get_time();
	   step();
	   end = get_time();
	   printf("Time to run: %f\n", end-start);
   }
   else
   {
	   step();
   }
   glutPostRedisplay();
}

void keyboardPress(unsigned char key, int x, int y)
{
	switch(key)
	{
		case 'a':
			posX+=MOVE_DISTANCE;
			break;
		case 'd':
			posX-=MOVE_DISTANCE;
			break;
		case 'w':
			posY-=MOVE_DISTANCE;
			break;
		case 's':
			posY+=MOVE_DISTANCE;
			break;
		case 'q':
			--scale;
			break;
		case 'e':
			++scale;
			break;
		case 'z':
			printf("\nCPU processing\n");
			processWithCPU();
			break;
		case 'x':
			printf("\nGPU Basic processing\n");
			processWithGPUBasic();
			break;
		case 'c':
			if(X_AXIS < 1024 && Y_AXIS < 1024)
			{
				printf("\nGPU Optimized processing\n");
				processWithGPUOpt();
			}
			else
			{
				printf("\nGPU Optimized unable to run. Cell count is too large\n");
			}
			break;
		case 't':
			timmingVisible ^=0x1;
			break;
	}
}

void printCommands()
{
	printf("-----Welcome to the game of life-----");
	printf("\n");
	printf("\n");
	printf("Arguments");
	printf("\n");
	printf("\n");
	printf("-c (default 200) This is a taken as the cells in an XxY so this is 200x200. This argument is and integer");
	printf("\n");
	printf("\n");
	printf("Controls");
	printf("\n");
	printf("\n");
	printf("w\t move screen up\n");
	printf("s\t move screen down\n");
	printf("a\t move screen left\n");
	printf("d\t move screen right\n");
	printf("\n");
	printf("q\t zoom out\n");
	printf("e\t zoom in\n");
	printf("\n");
	printf("z\t cpu implementation\n");
	printf("x\t gpu naïve implementation\n");
	printf("c\t gpu optimised implementation\n");
	printf("\n");
	printf("t\t turn on and off implementation timing\n");
	printf("\n");
	printf("\n");
	printf("Please read the Readme.txt as it explains the decreased improvement in the optimised version");
	printf("\n");
	printf("\n");
}

uint8_t getUserInput(int argc, char* argv[])
{
	if(argc == 1)
	{
		printf("\n\nUsing default inputs\n\n");
		return 0;
	}
	else if(argc > 3)
	{
		printf("\n\nInvalid input count");
		return 1;
	}
	uint32_t val = Y_AXIS;
	if(argv[1][0] != '-')
	{
		return 1;
	}
	switch (argv[1][1]) {
	case 'c':
		val = atoi(argv[2]);
		break;
	default:
		return 1;
	}

	Y_AXIS = val;
	X_AXIS = Y_AXIS;


	return 0;
}

int main(int argc, char* argv[])
{
	printCommands();
	if(getUserInput(argc,argv) == 1)
	{
		printf("\n\nInvalid input\n\n");
		exit(1);
	}
    glutInit (&argc,argv);
    glutInitDisplayMode(GLUT_DOUBLE| GLUT_RGB);
    glutInitWindowPosition(200, 200);
    glutInitWindowSize(800, 800);
    glutCreateWindow("-----The Game of Life-----");

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, X_AXIS, 0.0, Y_AXIS,-1000.0f, 1000.0f);
    glMatrixMode (GL_MODELVIEW);
    glColor3f(1.0,1.0,1.0);

    glutDisplayFunc(drawLines);
    glutTimerFunc(INTERVAL, update,0);
    glutKeyboardFunc(keyboardPress);

    GameLogic(X_AXIS, Y_AXIS) ;

    printf("\n\nStarting\n\n");
    glutMainLoop();
    printf("\n\nExiting\n\n");
    desructor_GameLogic();
    return (EXIT_SUCCESS);
}

