#include <stdlib.h>

// Fixed seed for standard random number generator
// to have same starting grid values.
// This is so we can compare running time from same
// starting grid for each of the versions of the code.
#define SRAND_VALUE 1985  
 

void init_ghosts(int *grid, int *newGrid, int dim) {
    int i;

    // ghost rows
    for (i = 1; i <= dim; i++) {
      // copy first row to bottom ghost row
      grid[(dim+2)*(dim+1)+i] = grid[(dim+2)+i];
      
      // copy last row to top ghost row
      grid[i] = grid[(dim+2)*dim + i];
    }

    // ghost columns
    for (i = 0; i <= dim+1; i++) {
      // copy first column to right most ghost column
      grid[i*(dim+2)+dim+1] = grid[i*(dim+2)+1];
      
      // copy last column to left most ghost column
      grid[i*(dim+2)] = grid[i*(dim+2) + dim];
    }
}

int apply_rules(int *grid, int *newGrid, int dim) {
    int i,j;
    int num_alive = 0;

    // iterate over the grid
    for (i = 1; i <= dim; i++) {
        for (j = 1; j <= dim; j++) {
            int id = i*(dim+2) + j;
            
            int numNeighbors = 
                grid[id+(dim+2)] + grid[id-(dim+2)]   // lower + upper
                + grid[id+1] + grid[id-1]             // right + left
                + grid[id+(dim+3)] + grid[id-(dim+3)] // diagonal lower + upper right
                + grid[id-(dim+1)] + grid[id+(dim+1)];// diagonal lower + upper left

            // the game rules
            if (grid[id] == 1 && numNeighbors < 2) {
                newGrid[id] = 0;
            } else if (grid[id] == 1 && (numNeighbors == 2 || numNeighbors == 3)) {
                newGrid[id] = 1;
                num_alive++;
            } else if (grid[id] == 1 && numNeighbors > 3) {
                newGrid[id] = 0;
            } else if (grid[id] == 0 && numNeighbors == 3) {
                newGrid[id] = 1;
                num_alive++;
            } else {
                newGrid[id] = grid[id];
                if (grid[id] == 1) num_alive++;
            }
        }
    }
    return num_alive;
}

void update_grid(int *grid, int *newGrid, int dim) {
    int i,j;

    // copy new grid over, as pointers cannot be switched on the device
    for(i = 1; i <= dim; i++) {
        for(j = 1; j <= dim; j++) {
            int id = i*(dim+2) + j;
            grid[id] = newGrid[id];
        }
    }
}

// 4 parts to one iteration split into functions
int gol(int *grid, int *newGrid, int dim)
{
  
  init_ghosts(grid, newGrid, dim);
  
  int sub_total = apply_rules(grid, newGrid, dim);

  update_grid(grid, newGrid, dim); 

  return sub_total;   
    
}

// initialize the grid with random values one time
// at the beginning
void init_grid(int *grid, int dim) {
  int i, j;
  // assign initial population randomly
    srand(SRAND_VALUE);
    for(i = 1; i <= dim; i++) {
        for(j = 1; j <= dim; j++) {
            grid[i*(dim+2)+j] = rand() % 2;
        }
    }
}


