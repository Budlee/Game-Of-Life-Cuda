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

#define Y_AXIS 500  //Y_AXIS%4 should be equal 3 for 9 pixels in cell (3 rows of pixels)
#define X_AXIS 500 //X_AXIS%4 should be equal 3 for 9 pixels in cell (3 columns of pixels)
#define BLOCK_SIZE 4.0
#define INTERVAL (1000 / 60)

float posX = 0.01, posY = -0.1, posZ = 0;

void drawLines (void) {
    glClear(GL_COLOR_BUFFER_BIT);
    glLoadIdentity();
    glTranslatef(posX,posY,posZ);
    uint64_t yIndex = Y_AXIS;
    uint64_t xIndex = 0;
    uint8_t *cellsLocal = getGameOfLifeState();
    while (yIndex != 0)
    {
        xIndex = 0;
        while (xIndex < X_AXIS)
        {
            if (cellsLocal[(yIndex * X_AXIS) + xIndex] != 0)
            {
                glColor3f(0.0, 1.0, 0.0);
            }
            else
            {
                glColor3f(1.0, 1.0, 1.0);
            }
            glRecti(xIndex, yIndex, xIndex+BLOCK_SIZE, yIndex+BLOCK_SIZE);
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
   double start ,end;
   start = get_time();
   step();
   end = get_time();
   printf("Time to run: %f\n", end-start);
   glutPostRedisplay();
}

void idleUpdate()
{
   step();
   glutPostRedisplay();
}

void keyboardPress(unsigned char key, int x, int y)
{
	switch(key)
	{
		case 'a':
			++posX;
			break;
		case 'd':
			--posX;
			break;
		case 'w':
			--posY;
			break;
		case 's':
			++posY;
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
			printf("\nGPU Optimized processing\n");
			processWithGPUOpt();
			break;
	}
}

int main(int argc, char** argv)
{
    glutInit (&argc,argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowPosition(200, 0);
    glutInitWindowSize(800, 600);
    glutCreateWindow("-----The Game of Life Idle-----");

    glViewport(0, 0, X_AXIS, Y_AXIS);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, X_AXIS, 0.0, Y_AXIS,0.0f, 1.0f);
    glMatrixMode (GL_MODELVIEW);
    glColor3f(1.0,1.0,1.0);

    glutDisplayFunc(drawLines);
    glutTimerFunc(INTERVAL, update,0);
    glutKeyboardFunc(keyboardPress);

    GameLogic(X_AXIS, Y_AXIS) ;

    glutMainLoop();
    desructor_GameLogic();
    return (EXIT_SUCCESS);
}

