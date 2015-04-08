#include "GameLogic.h"


void GameLogic(int64_t xIn, int64_t yIn)
{
	processorType = CPU_PROCESSOR;
    x = xIn;
    y = yIn;
    cellSwitch = 0;
    int64_t totalBlockCount = x*y;
    cells = (uint8_t*)malloc(sizeof (uint8_t)*(totalBlockCount));
    outputCell = (uint8_t*)malloc(sizeof (uint8_t)*(totalBlockCount));
    srand(time(NULL));
    while(totalBlockCount-- != 0)
    {
        if(rand()%2)
        {
            cells[totalBlockCount] = 1;
        }
        else
        {
        	cells[totalBlockCount] = 0;
        }
        outputCell[totalBlockCount]= 0;
    }
    cudaSetup();
}

void desructor_GameLogic()
{
	printf("\n\nEXIT\n\n");
	cudaFree(d_data_in);
	cudaFree(d_data_in_x);
	cudaFree(d_data_in_y);
	cudaFree(d_data_out);
    free(cells);
    free(outputCell);
}

uint8_t* getGameOfLifeState()
{
    return cells;
}

void processWithCPU(){processorType = CPU_PROCESSOR;}
void processWithGPUBasic(){processorType = GPU_BASIC_PROCESSOR;}
void processWithGPUOpt(){processorType = GPU_OPTIMIZED_PROCESSOR;}

void step()
{
	if(processorType == CPU_PROCESSOR)
	{
		cpuImplementation();
	}
	else
	{
		GPUImplementation();
	}
}

void cpuImplementation()
{
    uint64_t yId =0, xId = 0;
    while(yId < y)
    {
        xId = 0;
        while(xId < x)
        {
        	outputCell[xId+(yId*y)] = 0;
            uint8_t localCellCount =  surrondingCellCount(cells, xId, yId, x, y);
            switch(localCellCount)
            {
                case 2:
                    if(cells[xId+(yId*y)])
                    {
                    	outputCell[xId+(yId*y)] = 1;
                    }
                    break;
                case 3:
                	outputCell[xId+(yId*y)] = 1;
                    break;
            }
            ++xId;
        }
        ++yId;
    }
    uint8_t * switcher = cells;
    cells = outputCell;
    outputCell = switcher;
}

__host__ __device__ uint8_t surrondingCellCount(uint8_t *cellsLocal, int64_t xCell, int64_t yCell, int64_t x, int64_t y)
{
    //Parse left -> right, top -> bottom
    uint8_t cellCount = 0;
    xCell= xCellMinus(1,xCell, x);
    yCell = yCellMinus(1,yCell, y);
    uint8_t searchIndex = 0;
    while(searchIndex < 9)
    {
        if(searchIndex == 4)
        {
        	xCell = xCellPlus(1,xCell, x);
            ++searchIndex;
            continue;
        }
        if(searchIndex == 3 || searchIndex == 6)
        {
        	xCell =xCellMinus(3,xCell, x);
        	yCell = yCellPlus(1,yCell, y);
        }
        //find cell
        int64_t cell = xCell + (y * yCell);
        if(cellsLocal[cell])
        {
            cellCount++;
        }

        xCell = xCellPlus(1,xCell, x);
        ++searchIndex;
    }

    return cellCount;
}

__device__ uint8_t surrondingCellCountOptimized(uint8_t *cellsLocal, int64_t xCell, int64_t x, int64_t y)
{
	//Parse left -> right, top -> bottom
	    uint8_t cellCount = 0;
	    int64_t yCell = 1;
	    xCell= xCellMinus(1,xCell, x);
	    yCell = yCellMinus(1,yCell, y);
	    uint8_t searchIndex = 0;
	    while(searchIndex < 9)
	    {
	        if(searchIndex == 4)
	        {
	        	xCell = xCellPlus(1,xCell, x);
	            ++searchIndex;
	            continue;
	        }
	        if(searchIndex == 3 || searchIndex == 6)
	        {
	        	xCell =xCellMinus(3,xCell, x);
	        	yCell = yCellPlus(1,yCell, y);
	        }
	        //find cell
	        int64_t cell = xCell + (y * yCell);
	        if(cellsLocal[cell])
	        {
	            cellCount++;
	        }
	        if(cellCount > 3)
	        {
	        	return cellCount;
	        }

	        xCell = xCellPlus(1,xCell, x);
	        ++searchIndex;
	    }

	    return cellCount;

}

__host__ __device__ int64_t xCellPlus(uint8_t add, int64_t value, int64_t x)
{
    if(add == 0 && add < x)
    {
        return -1;
    }
    if(value+add >= x)
    {
        value=(value+add) - x ;
    }
    else
    {
        value+=add;
    }
    return value;
}

__host__ __device__ int64_t xCellMinus(uint8_t minus, int64_t value, int64_t x)
{
    if(minus == 0 && minus < x)
    {
        return -1;
    }
    if(value-minus < 0)
    {
        value = x + (value-minus);
    }
    else
    {
        value -= minus;
    }
    return value;
}

__host__ __device__ int64_t yCellPlus(uint8_t add, int64_t value, int64_t y)
{
    if(add == 0 && add < y)
    {
        return -1;
    }
    if(value+add >= y)
    {
        value=(value+add) - y;
    }
    else
    {
        value+=add;
    }
    return value;
}

__host__ __device__ int64_t yCellMinus(uint8_t minus, int64_t value, int64_t y)
{
    if(minus == 0 && minus < y)
    {
        return -1;
    }
    if(value-minus < 0)
    {
        value = y +(value-minus);
    }
    else
    {
        value -= minus;
    }
    return value;
}

void cudaSetup()
{
    printf("Get GPU properties\n");
    int64_t totalBlockCount = x*y;
	blockSize = 1024;
	gridSize = ((totalBlockCount + (blockSize-1))/blockSize);
	optBlockSize = x;
	optGridSize =(((x*y) + (optBlockSize-1))/optBlockSize);
	if((x >= 1024) || (y >= 1024))
	{
		printf("\n\nUnable to use optimized GPU version with x & y larger than 1023\n\n");
	}

    int devID;
    cudaDeviceProp props;
    cudaGetDevice(&devID);
    cudaGetDeviceProperties(&props, devID);
    printf("Allocate memory on GPU\n");
    cudaMalloc((void**) &d_data_in, (sizeof(uint8_t) * totalBlockCount));
    cudaMalloc((void**) &d_data_in_x, (sizeof(int64_t)));
    cudaMalloc((void**) &d_data_in_y, (sizeof(int64_t)));
    cudaMalloc((void**) &d_data_out, (sizeof(uint8_t) * totalBlockCount));

}

void GPUImplementation()
{
	//Copy data to GPU
	int64_t totalBlockCount = x*y;
    cudaMemcpy(d_data_in, cells, (sizeof(uint8_t) * totalBlockCount), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data_in_x, &x, sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data_in_y, &y, sizeof(int64_t), cudaMemcpyHostToDevice);
    //Executing GPU calculation
    if(processorType == GPU_OPTIMIZED_PROCESSOR && (x < 1024) && (y < 1024))
    {
		gpuGameOfLifeOptimized<<<optGridSize, optBlockSize, sizeof(uint8_t)*(x * 3)>>>(d_data_in, d_data_in_x, d_data_in_y, d_data_out);
    }
    else
    {
    	gpuGameOfLifeNaive<<<gridSize, blockSize>>>(d_data_in, d_data_in_x, d_data_in_y, d_data_out);
    }
    cudaThreadSynchronize();
    //Complete GPU calculation
    //Reading GPU data out
    cudaMemcpy(outputCell, d_data_out, (sizeof(uint8_t) * totalBlockCount), cudaMemcpyDeviceToHost);
    uint8_t * switcher = cells;
	cells = outputCell;
	outputCell = switcher;
}
/**
 * This optimization does not work... Also this has to be for grid sizes less than 1024 x and y access
 * 					The naieve call count is
 * 					read global memory calls: 		9
 * 					write global memory calls: 		2
 * 					read shared memory calls:		0
 * 					write shared memory calls:		0
 *
 * 					The optimized call count is
 * 					read global memory calls: 		3
 * 					write global memory calls: 		1
 * 					read shared memory calls:		12
 * 					write shared memory calls:		3
 *
 */
__global__ void gpuGameOfLifeOptimized(uint8_t* cellsGlobal, int64_t* dataX, int64_t* dataY, uint8_t *outputCellLocal)
{
	extern __shared__ uint8_t sData[];
	uint64_t tId = threadIdx.x + (blockIdx.x * blockDim.x);
	uint64_t x = *dataX, y = *dataY;
	uint64_t totalBlockCount = x*y;
	uint64_t yId = tId/y;
	uint64_t xId = tId - (yId*x);
	//Pass data to shared
	if(tId <= totalBlockCount)
	{
		uint8_t rowIndex = 0;
		uint64_t yVal = yCellMinus(1,yId,y);
		while (rowIndex < 3)
		{
			sData[(rowIndex * x) + xId] =cellsGlobal[((yVal*x)) + xId];
			yVal = yCellPlus(1,yVal, y);
			++rowIndex;
		}
	}
	syncthreads();
	if(tId <= totalBlockCount)
	{
		uint8_t localCellCount =  surrondingCellCountOptimized(sData, xId, x, y);
		uint64_t result = 0;
		switch(localCellCount)
		{
			case 2:
				if(sData[x + xId])
				{
					result = 1;
				}
				break;
			case 3:
				result = 1;
				break;
		}
		outputCellLocal[xId+(yId*x)] = result;
	}
}

__global__ void gpuGameOfLifeNaive(uint8_t* cellsLocal, int64_t* dataX, int64_t* dataY, uint8_t *outputCellLocal)
{

	uint64_t tId = threadIdx.x + (blockIdx.x * blockDim.x);
	uint64_t x = *dataX, y = *dataY;
	uint64_t totalBlockCount = x*y;
	if(tId <= totalBlockCount)
	{
		uint64_t yId = tId/y;
		uint64_t xId = tId - (yId*x);
		outputCellLocal[xId+(yId*x)] = 0;
		uint8_t localCellCount =  surrondingCellCount(cellsLocal, xId, yId, x, y);
		switch(localCellCount)
		{
			case 2:
				if(cellsLocal[xId+(yId*x)])
				{
					outputCellLocal[xId+(yId*x)] = 1;
				}
				break;
			case 3:
				outputCellLocal[xId+(yId*x)] = 1;
				break;
		}
	}
}

//Improved Naieve
//__global__ void gpuGameOfLifeNaive(uint8_t* cellsLocal, int64_t* dataX, int64_t* dataY, uint8_t *outputCellLocal)
//{
//	int64_t x = *dataX, y = *dataY;
//	int64_t xId = 0;
//	int64_t yIndex = threadIdx.x + (blockIdx.x * blockDim.x);
//	int64_t totalBlockCount = x*y;
//	if(yIndex <= totalBlockCount)
//	{
//		xId = 0;
//		while(xId < x)
//		{
//			outputCellLocal[xId+(yIndex*y)] = 0;
//			uint8_t localCellCount =  surrondingCellCount(cellsLocal, xId, yIndex, x, y);
//			switch(localCellCount)
//			{
//				case 2:
//					if(cellsLocal[xId+(yIndex*y)])
//					{
//						outputCellLocal[xId+(yIndex*y)] = 1;
//					}
//					break;
//				case 3:
//					outputCellLocal[xId+(yIndex*y)] = 1;
//					break;
//			}
//
//			++xId;
//		}
//	}
//}

//Origonal
//__global__ void gpuGameOfLifeNaive(uint8_t* cellsLocal, int64_t* dataX, int64_t* dataY, uint8_t *outputCellLocal)
//{
//	int64_t x = *dataX, y = *dataY;
//	int64_t yId =0, xId = 0;
//	int64_t ysId = threadIdx.x + (blockIdx.x * blockDim.x);
//	if(ysId == 1)
//	{
//	    while(yId < y)
//	    {
//	        xId = 0;
//	        while(xId < x)
//	        {
//	        	outputCellLocal[xId+(yId*y)] = 0;
//	            uint8_t localCellCount =  surrondingCellCount(cellsLocal, xId, yId, x, y);
//				switch(localCellCount)
//				{
//					case 2:
//						if(cellsLocal[xId+(yId*y)])
//						{
//							outputCellLocal[xId+(yId*y)] = 1;
//						}
//						break;
//					case 3:
//						outputCellLocal[xId+(yId*y)] = 1;
//						break;
//				}
//
//	            ++xId;
//	        }
//	        ++yId;
//	    }
//	}
//}
