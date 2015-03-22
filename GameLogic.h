/*
 * File:   GameLogic.h
 * Author: matt
 *
 * Created on 16 March 2015, 10:45
 */

#ifndef GAMELOGIC_H
#define	GAMELOGIC_H

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

#define CPU_PROCESSOR 0
#define GPU_BASIC_PROCESSOR 1
#define GPU_OPTIMIZED_PROCESSOR 2

#ifdef	__cplusplus
extern "C" {
#    endif

    void GameLogic(int64_t, int64_t);
    void desructor_GameLogic();
    uint8_t* getGameOfLifeState();
    void step();

    void cpuImplementation();

    static uint8_t processorType;
    void processWithCPU();
    void processWithGPUBasic();
    void processWithGPUOpt();
    void cudaSetup();
    void naiveGPUImplementation();

    __global__ void gpuGameOfLifeNaive(uint8_t* cellsLocal, int64_t* dataX, int64_t* dataY, uint8_t *outputCell);
    __host__ __device__ uint8_t surrondingCellCount(uint8_t *cells, int64_t xCell, int64_t yCell, int64_t x, int64_t y);
    __host__ __device__    int64_t xCellPlus(uint8_t add, int64_t value, int64_t x);
    __host__ __device__    int64_t xCellMinus(uint8_t minus, int64_t value, int64_t x);
    __host__ __device__    int64_t yCellPlus(uint8_t add, int64_t value, int64_t y);
    __host__ __device__    int64_t yCellMinus(uint8_t minus, int64_t value, int64_t y);

    static int64_t x,y;
    static uint8_t *cells, *outputCell;

    static uint8_t *d_data_in, *d_data_out;
    static int64_t *d_data_in_x, *d_data_in_y;

    static uint32_t blockSize, gridSize;
    static uint8_t cellSwitch;


#    ifdef	__cplusplus
}
#endif

#endif	/* GAMELOGIC_H */

