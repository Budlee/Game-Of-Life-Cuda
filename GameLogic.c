


#include <stdio.h>

#include "GameLogic.h"


void GameLogic(int64_t xIn, int64_t yIn)
{
    x = xIn;
    y = yIn;
    int64_t totalBlockCount = x*y;
    cells = (uint8_t*)malloc(sizeof (uint8_t)*(totalBlockCount));
    outputCell = (uint8_t*)malloc(sizeof (uint8_t)*(totalBlockCount));
    totalBlockCount = totalBlockCount/2;
    while(totalBlockCount-- != 0)
    {
        if(rand()%2)
        {
            cells[totalBlockCount] = 1;
        }
    }
}

void desructor_GameLogic()
{
    free(cells);
    free(outputCell);
}

uint8_t* getGameOfLifeState()
{
    return cells;
}

void step()
{
    cpuImplementation();
}

void cpuImplementation()
{
    uint64_t yId =0, xId = 0;
    while(yId < y)
    {
        xId = 0;
        while(xId < x)
        {
            uint8_t localCellCount =  surrondingCellCount(xId, yId);
            
            switch(localCellCount)
            {
                case 0:
                case 1:
                    outputCell[xId+(yId*y)] = 0;
                    break;
                case 2:
                    if(cells[xId+(yId*y)])
                    {
                        outputCell[xId+(yId*y)] = 1;
                    }
                    break;
                case 3:
                    outputCell[xId+(yId*y)] = 1;
                    break;
                default:
                    outputCell[xId+(yId*y)] = 0;
                    break; 
            }
            ++xId;
        }
        ++yId;
    }
    uint64_t temp = x*y;
    yId = 0;
    while(yId < temp)
    {
        cells[yId] = outputCell[yId];
        ++yId;
    }
}

uint8_t surrondingCellCount(int64_t xCell, int64_t yCell)
{
    //Parse left -> right, top -> bottom
    uint8_t cellCount = 0;
    xCellMinus(1,&xCell);
    yCellMinus(1,&yCell);
    uint8_t searchIndex = 0;
    while(searchIndex < 9)
    {
        if(searchIndex == 4)
        {
            xCellPlus(1,&xCell);
            ++searchIndex;
            continue;
        }
        if(searchIndex % 3 == 0 && searchIndex != 0)
        {
            xCellMinus(3,&xCell);
            yCellPlus(1,&yCell);
        }
        
        //find cell
        int64_t cell = xCell + (y*yCell);
        
        if(cells[cell])
        {
            cellCount++;
        }
        
        xCellPlus(1,&xCell);
        ++searchIndex;
    }
    
    return cellCount;
}

void xCellPlus(uint8_t add, int64_t* value)
{
    if(add == 0 && add < x)
    {
        return;
    }
    if(*value+add >= x)
    {
        *value=(*value+add) - x ;
    }
    else
    {
        *value+=add;
    }
}

void xCellMinus(uint8_t minus, int64_t* value)
{
    if(minus == 0 && minus < x)
    {
        return;
    }
    if(*value-minus < 0)
    {
        *value = x + (*value-minus);
    }
    else
    {
        *value -= minus;
    }
}

void yCellPlus(uint8_t add, int64_t* value)
{
    if(add == 0 && add < y)
    {
        return;
    }
    if(*value+add >= y)
    {
        *value=(*value+add) - y;
    }
    else
    {
        *value+=add;
    }
}

void yCellMinus(uint8_t minus, int64_t* value)
{
    if(minus == 0 && minus < y)
    {
        return;
    }
    if(*value-minus < 0)
    {
        *value = y +(*value-minus);
    }
    else
    {
        *value -= minus;
    }
}