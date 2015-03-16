/* 
 * File:   GameLogic.h
 * Author: matt
 *
 * Created on 16 March 2015, 10:45
 */

#ifndef GAMELOGIC_H
#define	GAMELOGIC_H

#include <stdlib.h>
#include <inttypes.h>

#ifdef	__cplusplus
extern "C" {
#    endif

    void GameLogic(int64_t, int64_t);
    void desructor_GameLogic();
    uint8_t* getGameOfLifeState();
    void step();
    
    void cpuImplementation();
    
    
    uint8_t surrondingCellCount(int64_t xLocal, int64_t yLocal);
    void xCellPlus(uint8_t add, int64_t *value);
    void xCellMinus(uint8_t minus, int64_t *value);
    void yCellPlus(uint8_t add, int64_t *value);
    void yCellMinus(uint8_t minus, int64_t* value);
    
    static int64_t x,y;
    uint8_t *cells, *outputCell;


#    ifdef	__cplusplus
}
#endif

#endif	/* GAMELOGIC_H */

