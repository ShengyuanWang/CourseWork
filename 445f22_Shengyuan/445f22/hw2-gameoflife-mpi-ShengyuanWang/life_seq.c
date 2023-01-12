/* DESCRIPTION: Parallel code for simulating a cellular automaton running 
 *                  Conway's Game of Life.
 * AUTHOR:      Aaron Weeden, Shodor Education Foundation, Inc.
 * DATE:        January 2012
 *
 * Updated by Libby Shoop, Macalester College
 */

/***********************
 * Libraries to import *
 ***********************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "seq_time.h"  // Libby's timing function that is similar to omp style

#define ALIVE 1
#define DEAD 0

/********************************************
 * Need at least this many rows and columns *
 ********************************************/
const int MINIMUM_ROWS = 1;
const int MINIMUM_COLUMNS = 1;
const int MINIMUM_TIME_STEPS = 1;

/*****************************************************
 * Add an "s" to the end of a value's name if needed *
 *****************************************************/
void pluralize_value_if_needed(int value)
{
    if(value != 1)
        fprintf(stderr, "s");

    return;
}

/*******************************************************************************
 * Make sure a value is >= another value, print error and return -1 if it isn't
 ******************************************************************************/
int assert_minimum_value(char which_value[16], int actual_value,
        int expected_value)
{
    int retval;

    if(actual_value < expected_value)
    {
        fprintf(stderr, "ERROR: %d %s", actual_value, which_value);
        pluralize_value_if_needed(actual_value);
        fprintf(stderr, "; need at least %d %s", expected_value, which_value);
        pluralize_value_if_needed(expected_value);
        fprintf(stderr, "\n");
        retval = -1;
    }
    else
        retval = 0;

    return retval;
}

/******************************************************************************
 * Print a function name and exit if the specified boolean expression is true *
 ******************************************************************************/
void exit_if(int boolean_expression, char function_name[32])
{
    if(boolean_expression)
    {
        fprintf(stderr, "ERROR in %s\n", function_name);
        exit(-1);
    }

    return;
}

// debug print function defined at end of file
void printGrid(int **our_current_grid, 
               int NUMBER_OF_ROWS, int NUMBER_OF_COLUMNS);

/****************
 * Main program *
 ****************/
int main(int argc, char **argv)
{
    // default values updated by command line arguments
    int NUMBER_OF_ROWS = 5;
    int NUMBER_OF_COLUMNS = 5;
    int NUMBER_OF_TIME_STEPS = 5; 
    
    // used for updating new value based on my neighbor's values
    int our_current_row, my_current_column;
    int my_neighbor_row, my_neighbor_column;
    int my_number_of_alive_neighbors;

    int c;   // for command line argument parsing
    int return_value; //for checking if row, col, time step values are sensible

    // the two 2D grids of integers
    int **our_current_grid;
    int **our_next_grid;

    // keeping track of the time step
    int current_time_step;
 

    /* 1.  Parse command line arguments */ 
    while ((c = getopt(argc, argv, "r:c:t:")) != -1)
    {
        switch (c)
        {
            case 'r':
                NUMBER_OF_ROWS = atoi(optarg);
                break;
            case 'c':
                NUMBER_OF_COLUMNS = atoi(optarg);
                break;
            case 't':
                NUMBER_OF_TIME_STEPS = atoi(optarg);
                break;
            case '?':
            default:

                fprintf(stderr, "Usage: %s [-r NUMBER_OF_ROWS] [-c NUMBER_OF_COLUMNS] [-t NUMBER_OF_TIME_STEPS]\n", argv[0]);

                exit(-1);
        }
    }
    argc -= optind;
    argv += optind;

    /* 2.  Make sure we have enough rows, columns, and time steps */
    return_value = assert_minimum_value("row", NUMBER_OF_ROWS, MINIMUM_ROWS);
    return_value += assert_minimum_value("column", NUMBER_OF_COLUMNS,
            MINIMUM_COLUMNS);
    return_value += assert_minimum_value("time step", NUMBER_OF_TIME_STEPS,
            MINIMUM_TIME_STEPS);

    /* 3.  Exit if we don't */
    if (return_value != 0)
        exit(-1);

    double start = c_get_wtime();  //////// start timing

    /* 4.  Allocate enough space in our current grid and next grid for the
     *  number of rows and the number of columns, plus the ghost rows
     *  and columns */
    exit_if(((our_current_grid = (int**)malloc((NUMBER_OF_ROWS + 2) 
                        * (NUMBER_OF_COLUMNS + 2) * sizeof(int))) == NULL),
            "malloc(our_current_grid)");
    exit_if(((our_next_grid = (int**)malloc((NUMBER_OF_ROWS + 2) 
                        * (NUMBER_OF_COLUMNS + 2) * sizeof(int))) == NULL),
            "malloc(our_next_grid)");
    for (our_current_row = 0; our_current_row <= NUMBER_OF_ROWS + 1;
            our_current_row++)
    {
        exit_if(((our_current_grid[our_current_row]
                        = (int*)malloc((NUMBER_OF_COLUMNS + 2) * sizeof(int))) 
                    == NULL), "malloc(our_current_grid[some_row])");
        exit_if(((our_next_grid[our_current_row]
                        = (int*)malloc((NUMBER_OF_COLUMNS + 2) * sizeof(int)))
                    == NULL), "malloc(our_next_grid[some_row])");
    }

    /* 5.  Initialize the grid (each cell gets a random state) */
    for (our_current_row = 1; our_current_row <= NUMBER_OF_ROWS;
            our_current_row++)
    {
        for(my_current_column = 1; my_current_column <= NUMBER_OF_COLUMNS;
                my_current_column++)
        {
            our_current_grid[our_current_row][my_current_column] =
                random() % (ALIVE + 1);
        }
    }

    /* 6.  Run the simulation for the specified number of time steps */
    for(current_time_step = 0; current_time_step <= NUMBER_OF_TIME_STEPS - 1;
            current_time_step++)
    {
        /* 6.1 set up the ghost rows */
        for (my_current_column = 0;
                my_current_column <= NUMBER_OF_COLUMNS + 1;
                my_current_column++)
        {
            /* 6.1.1.  Set our top row to be the same as our second-to-last 
             *  row */
            our_current_grid[0][my_current_column]
                = our_current_grid[NUMBER_OF_ROWS][my_current_column];

            /* 6.1.2.  Set our bottom row to be the same as our 
             *  second-to-top row */
            our_current_grid[NUMBER_OF_ROWS + 1][my_current_column]
                = our_current_grid[1][my_current_column];
        }

        /* 6.2.  Set up the ghost columns */
        for (our_current_row = 0; our_current_row <= NUMBER_OF_ROWS + 1;
                our_current_row++)
        {
            /* 6.2.1.  The left ghost column is the same as the farthest-right,
             *  non-ghost column */
            our_current_grid[our_current_row][0] =
                our_current_grid[our_current_row][NUMBER_OF_COLUMNS];

            /* 6.2.2.  The right ghost column is the same as the farthest-left,
             *  non-ghost column */
            our_current_grid[our_current_row][NUMBER_OF_COLUMNS + 1] =
                our_current_grid[our_current_row][1];
        }

        /* 6.3 DEBUG: If define flag set in Makefile, Display current grid */
#ifdef SHOW_RESULTS
 
        printf("Time Step %d:\n", current_time_step);
        printGrid(our_current_grid, NUMBER_OF_ROWS, NUMBER_OF_COLUMNS);
 
#endif

        /* 6.4.  Determine our next grid -- for each row, do the following: */
        for (our_current_row = 1; our_current_row <= NUMBER_OF_ROWS;
                our_current_row++)
        {
            /* For each column, do the following: */

            for (my_current_column = 1; my_current_column <= NUMBER_OF_COLUMNS;
                    my_current_column++)
            {
                /* 6.4.1.  Initialize the count of ALIVE neighbors to 0 */
                my_number_of_alive_neighbors = 0;

                /* 6.4.2.  For each row of the cell's neighbors, do the
                 *  following: */
                for (my_neighbor_row = our_current_row - 1;
                        my_neighbor_row <= our_current_row + 1;
                        my_neighbor_row++)
                {
                    /* X.D.1.b.i.  For each column of the cell's neighbors, do
                     *  the following: */
                    for (my_neighbor_column = my_current_column - 1;
                            my_neighbor_column <= my_current_column + 1;
                            my_neighbor_column++)
                    {
                        /* X.D.1.b.i.I.  If the neighbor is not the cell itself,
                         *  and the neighbor is ALIVE, do the following: */
                        if ((my_neighbor_row != our_current_row
                                    || my_neighbor_column != my_current_column)
                                && (our_current_grid[my_neighbor_row]
                                    [my_neighbor_column] == ALIVE))
                        {
                            /* X.D.1.b.i.I.A.  Add 1 to the count of the 
                             *  number of ALIVE neighbors */
                            my_number_of_alive_neighbors++;
                        }
                    }
                }

                /* X.D.1.c.  Apply Rule 1 of Conway's Game of Life */
                if (my_number_of_alive_neighbors < 2)
                {
                    our_next_grid[our_current_row][my_current_column] = DEAD;
                }

                /* X.D.1.d.  Apply Rule 2 of Conway's Game of Life */
                if (our_current_grid[our_current_row][my_current_column] == ALIVE
                        && (my_number_of_alive_neighbors == 2
                            || my_number_of_alive_neighbors == 3))
                {
                    our_next_grid[our_current_row][my_current_column] = ALIVE;
                }

                /* X.D.1.e.  Apply Rule 3 of Conway's Game of Life */
                if (my_number_of_alive_neighbors > 3)
                {
                    our_next_grid[our_current_row][my_current_column] = DEAD;
                }

                /* X.D.1.f.  Apply Rule 4 of Conway's Game of Life */
                if (our_current_grid[our_current_row][my_current_column] == DEAD
                        && my_number_of_alive_neighbors == 3)
                {
                    our_next_grid[our_current_row][my_current_column] = ALIVE;
                }
            }
        }

        /* 6.5.  copy the next grid into the current grid */
        for (our_current_row = 1; our_current_row <= NUMBER_OF_ROWS;
                our_current_row++)
        {
            for (my_current_column = 1; my_current_column <= NUMBER_OF_COLUMNS;
                    my_current_column++)
            {
                our_current_grid[our_current_row][my_current_column] =
                    our_next_grid[our_current_row][my_current_column];
            }
        }
    }   // END OF SIMULATION LOOP (step 6)

    double end = c_get_wtime();
    printf("time: %f seconds\n", end - start);

    /*  7. Deallocate data structures */
    for (our_current_row = NUMBER_OF_ROWS + 1; our_current_row >= 0;
            our_current_row--)
    {
        free(our_next_grid[our_current_row]);
        free(our_current_grid[our_current_row]);
    }
    free(our_next_grid);
    free(our_current_grid);

    return 0;
}

/* for debugging, print the grid */

void printGrid(int **our_current_grid, 
               int NUMBER_OF_ROWS, int NUMBER_OF_COLUMNS) {

    int our_current_row;
    int my_current_column;

    for (our_current_row = 0; our_current_row <= NUMBER_OF_ROWS + 1;
            our_current_row++)
    {
        if (our_current_row == 1)
        {
            for (my_current_column = 0;
                    my_current_column <= NUMBER_OF_COLUMNS + 1 + 2;
                    my_current_column++)
            {
                printf("- ");
            }
            printf("\n");
        }

        for (my_current_column = 0;
                my_current_column <= NUMBER_OF_COLUMNS + 1; 
                my_current_column++)
        {
            if (my_current_column == 1)
            {
                printf("| ");
            }

            printf("%d ", our_current_grid[our_current_row]
                    [my_current_column]);

            if (my_current_column == NUMBER_OF_COLUMNS)
            {
                printf("| ");
            }
        }
        printf("\n");

        if (our_current_row == NUMBER_OF_ROWS)
        {
            for (my_current_column = 0;
                    my_current_column <= NUMBER_OF_COLUMNS + 1 + 2;
                    my_current_column++)
            {
                printf("- ");
            }
            printf("\n");
        }
    }
    printf("\n");
}