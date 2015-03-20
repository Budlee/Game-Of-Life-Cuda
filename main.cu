/*
 * File:   main.c
 * Author: matt
 *
 * Created on 16 March 2015, 10:42
 */

#include <stdio.h>
#include <stdlib.h>

#include <GL/glew.h>
#include <GL/glu.h>
#include <GL/glut.h>



#include "GameLogic.h"

#define Y_AXIS 500  //Y_AXIS%4 should be equal 3 for 9 pixels in cell (3 rows of pixels)
#define X_AXIS 500 //X_AXIS%4 should be equal 3 for 9 pixels in cell (3 columns of pixels)
#define BLOCK_SIZE 4.0
#define INTERVAL (1000 / 60)

void drawLines (void) {
    glClear(GL_COLOR_BUFFER_BIT);
    glLoadIdentity();
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

void update(int value) {
   // Call update() again in 'interval' milliseconds
   glutTimerFunc(INTERVAL, update, 0);
   step();
   glutPostRedisplay();
}

void idleUpdate()
{
   step();
   glutPostRedisplay();
}

int main(int argc, char** argv)
{
    glutInit (&argc,argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowPosition(200, 0);
    glutInitWindowSize(X_AXIS, Y_AXIS);
    glutCreateWindow("-----The Game of Life Idle-----");


    glViewport(0, 0, X_AXIS, Y_AXIS);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, X_AXIS, 0.0, Y_AXIS,0.0f, 1.0f);
    glMatrixMode (GL_MODELVIEW);
    glColor3f(1.0,1.0,1.0);


    glutDisplayFunc(drawLines);
    glutTimerFunc(INTERVAL, update,0);
    //glutIdleFunc(idleUpdate);

    GameLogic(X_AXIS, Y_AXIS) ;


    glutMainLoop();
    //desructor_GameLogic();
    return (EXIT_SUCCESS);
}

