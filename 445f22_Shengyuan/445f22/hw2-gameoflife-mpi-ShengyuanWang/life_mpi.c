/* DESCRIPTION: Parallel code for simulating a cellular automaton running 
*                  Conway's Game of Life.
* AUTHOR:      Aaron Weeden, Shodor Education Foundation, Inc.
* DATE:        January 2012
*
* Updated by Libby Shoop, Macalester College
*    Made certain each process uses a different random starting point.
*    Added ability to print for debugging with samll grids.
*/

/***********************
* Libraries to import *
***********************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

#include <mpi.h>

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
void exit_if(int boolean_expression, char function_name[32], int OUR_RANK)
{
  if(boolean_expression)
  {
      fprintf(stderr, "Rank %d ", OUR_RANK);

      fprintf(stderr, "ERROR in %s\n", function_name);
      exit(-1);
  }

  return;
}

// for debugging only
void mk_subgrid_copy(int **our_current_grid, int *grid_copy,
              int OUR_NUMBER_OF_ROWS, int NUMBER_OF_COLUMNS);

// debug print function defined at end of file
void printGrid(int *our_current_grid, 
              int OUR_NUMBER_OF_ROWS, int NUMBER_OF_COLUMNS); 

/*******************************************
* Main program that each process executes *
*******************************************/
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
  int return_value;  //for checking if row, col, time step values are sensible
  
  // the two 2D grids of integers
  int **our_current_grid;
  int **our_next_grid;

  // keeping track of the time step
  int current_time_step;

  // -------  MPI new values  -----------------------------------------------
  int OUR_NUMBER_OF_ROWS = 5;    // rows per process
  int OUR_RANK = 0;              // set when MPI_Init called
  int NUMBER_OF_PROCESSES = 1;   // set when invoke mpirun with -np
  int next_lowest_rank;          // neighbor process with subgrid 'above' us 
  int next_highest_rank;         // neighbor process with subgrid 'below' us

  int CONDUCTOR = 0;
  double startTime =0.0, totalTime= 0.0;

  int constantSeed = 0;
  // -------------------------------------------------------------------------
  MPI_Status status;
  /* 0.  Initialize the distributed memory environment (MPI)*/
  exit_if((MPI_Init(&argc, &argv) != MPI_SUCCESS), "MPI_Init", OUR_RANK);
  exit_if((MPI_Comm_rank(MPI_COMM_WORLD, &OUR_RANK) != MPI_SUCCESS),
          "MPI_Comm_rank", OUR_RANK);
  exit_if((MPI_Comm_size(MPI_COMM_WORLD, &NUMBER_OF_PROCESSES)
              != MPI_SUCCESS), "MPI_Comm_size", OUR_RANK);

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

              fprintf(stderr, "Usage: mpirun -np NUMBER_OF_PROCESSES %s [-r NUMBER_OF_ROWS] [-c NUMBER_OF_COLUMNS] [-t NUMBER_OF_TIME_STEPS]\n", argv[0]);

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

  /* TODO: 4. start timing, using a barrier 
  *  See patternlet for timing with barrier.
  */

  MPI_Barrier(MPI_COMM_WORLD);
  if (OUR_RANK == CONDUCTOR) {
    startTime = MPI_Wtime();
  }
  

  /* 5.  Determine our number of rows */
  OUR_NUMBER_OF_ROWS = NUMBER_OF_ROWS / NUMBER_OF_PROCESSES;
  if (OUR_RANK == NUMBER_OF_PROCESSES - 1)
  {
      OUR_NUMBER_OF_ROWS += NUMBER_OF_ROWS % NUMBER_OF_PROCESSES;
  }

  /* 6.  Allocate enough space in our current grid and next grid for the
    *  number of rows and the number of columns, plus the ghost rows
    *  and columns */
  exit_if(((our_current_grid = (int**)malloc((OUR_NUMBER_OF_ROWS + 2) 
                      * (NUMBER_OF_COLUMNS + 2) * sizeof(int))) == NULL),
          "malloc(our_current_grid)", OUR_RANK);
  exit_if(((our_next_grid = (int**)malloc((OUR_NUMBER_OF_ROWS + 2) 
                      * (NUMBER_OF_COLUMNS + 2) * sizeof(int))) == NULL),
          "malloc(our_next_grid)", OUR_RANK);
  for (our_current_row = 0; our_current_row <= OUR_NUMBER_OF_ROWS + 1;
          our_current_row++)
  {
      exit_if(((our_current_grid[our_current_row]
                      = (int*)malloc((NUMBER_OF_COLUMNS + 2) * sizeof(int))) 
                  == NULL), "malloc(our_current_grid[some_row])", OUR_RANK);
      exit_if(((our_next_grid[our_current_row]
                      = (int*)malloc((NUMBER_OF_COLUMNS + 2) * sizeof(int)))
                  == NULL), "malloc(our_next_grid[some_row])", OUR_RANK);
  }

  // for debugging, the conductor will receive and print the grid of each process,
  // so allocate twmporary space for this on each process.
#ifdef SHOW_RESULTS
  // For debugging, be sure to use a grid number of rows that is evenly
  // divisible by the number of processes. Then this temporary 1D array,
  // flattening the 2D version, will be the correct size for printing each  
  // subgrid for each process.
  int tmp_subgrid[(OUR_NUMBER_OF_ROWS+2) * (NUMBER_OF_COLUMNS+2)];
  constantSeed = 1;    // only with debug printing will we use a constant seed
#endif

  /* 7.  Initialize the grid (each cell gets a random state) */
  if (constantSeed) {
    srandom(OUR_RANK);  // constant, but different for each process
  } else {
    srandom(time(NULL) + OUR_RANK); // varies and different for each process
  }

  for (our_current_row = 1; our_current_row <= OUR_NUMBER_OF_ROWS;
          our_current_row++)
  {
      for (my_current_column = 1; my_current_column <= NUMBER_OF_COLUMNS;
              my_current_column++)
      {
          our_current_grid[our_current_row][my_current_column] =
              random() % (ALIVE + 1);
      }
  }

  /* 8.  Determine the process with the next-lowest rank */
  if (OUR_RANK == 0)
      next_lowest_rank = NUMBER_OF_PROCESSES - 1;
  else
      next_lowest_rank = OUR_RANK - 1;

  /* 9.  Determine the process with the next-highest rank */
  if (OUR_RANK == NUMBER_OF_PROCESSES - 1)
      next_highest_rank = 0;
  else
      next_highest_rank = OUR_RANK + 1;

  /* 10.  Run the simulation for the specified number of time steps */
  for (current_time_step = 0; current_time_step <= NUMBER_OF_TIME_STEPS - 1;
          current_time_step++)
  {
      /* 10.1.  Set up the ghost rows */

      /* 10.1.1.  Send our second-from-the-top row to the process with the
        *  next-lowest rank */
      exit_if((MPI_Send(our_current_grid[1], NUMBER_OF_COLUMNS + 2,
                      MPI_INT, next_lowest_rank, 0, MPI_COMM_WORLD) !=
                  MPI_SUCCESS),
              "MPI_Send(top row)", OUR_RANK);


      /* TODO 10.1.2.  Send our second-from-the-bottom row to the process 
        *  with the next-highest rank */
      exit_if((MPI_Send(our_current_grid[OUR_NUMBER_OF_ROWS], NUMBER_OF_COLUMNS + 2,
                      MPI_INT, next_highest_rank, 0, MPI_COMM_WORLD) !=
                  MPI_SUCCESS),
              "MPI_Send(bottom row)", OUR_RANK);
            

      /* TODO 10.1.3.  Receive our bottom row from the process with the 
        *  next-highest rank */
      exit_if((MPI_Recv(our_current_grid[OUR_NUMBER_OF_ROWS+1], NUMBER_OF_COLUMNS + 2,
                      MPI_INT, next_highest_rank, 0, MPI_COMM_WORLD, &status) !=
                  MPI_SUCCESS),
              "MPI_Recv(bottom row)", OUR_RANK);

      /* TODO 10.1.4.  Receive our top row from the process with the
        *  next-lowest rank */
      exit_if((MPI_Recv(our_current_grid[0], NUMBER_OF_COLUMNS + 2,
                      MPI_INT, next_lowest_rank, 0, MPI_COMM_WORLD, &status) !=
                  MPI_SUCCESS),
              "MPI_Recv(top row)", OUR_RANK);     

      /* 10.2.  Set up the ghost columns */
      for (our_current_row = 0; our_current_row <= OUR_NUMBER_OF_ROWS + 1;
              our_current_row++)
      {
          /* 10.2.1.  The left ghost column is the same as the farthest-right,
            *  non-ghost column */
          our_current_grid[our_current_row][0] =
              our_current_grid[our_current_row][NUMBER_OF_COLUMNS];

          /* 10.2.2.  The right ghost column is the same as the farthest-left,
            *  non-ghost column */
          our_current_grid[our_current_row][NUMBER_OF_COLUMNS + 1] =
              our_current_grid[our_current_row][1];
      }

#ifdef SHOW_RESULTS
       /* 10.3 Display grid */
//         Printing from all different processes gets interleaved and 
//        hard to decipher.
//        So we use a technique that is useful for debugging, but not for the
//        final working code whose scalability we will test.
//        Each rank sends its subgrid to the conductor process, who then prints it.
//        Study this code to see if you can determine how it is able to do this in
//        process order. Note that only the conductor is printing.
//        Not Something that you want to do when the arrays get large, 
//        but useful for debugging.
//   
    MPI_Barrier(MPI_COMM_WORLD);
    int token = current_time_step;
    int received = 0;              // token received
    MPI_Status status;
    if ( OUR_RANK == CONDUCTOR ) {                              
        mk_subgrid_copy(our_current_grid, tmp_subgrid, OUR_NUMBER_OF_ROWS, NUMBER_OF_COLUMNS);
        printf("Conductor subgrid iteration %d.\n", current_time_step);
        printGrid(tmp_subgrid, OUR_NUMBER_OF_ROWS, NUMBER_OF_COLUMNS);

        for (int next_pid = 1; next_pid < NUMBER_OF_PROCESSES; next_pid++) {
          MPI_Send(&token,              //  msg sent: token used to coordinate
                    1,                  //  size is single int 
                    MPI_INT,            //  type
                    next_pid,           //  destination
                    1,                  //  tag
                    MPI_COMM_WORLD);    //  communicator

          MPI_Recv(&tmp_subgrid,         //  msg received is subgrid
                    (OUR_NUMBER_OF_ROWS + 2) * (NUMBER_OF_COLUMNS + 2),  //  size
                    MPI_INT,             //  type
                    next_pid,            //  sender
                    2,                    //  tag
                    MPI_COMM_WORLD,       //  communicator
                    &status);             //  recv status

          printf("Process #%d of %d received subgrid from %d on iteration %d:\n", // show msg
                  OUR_RANK, NUMBER_OF_PROCESSES, next_pid, current_time_step);
          printGrid(tmp_subgrid, 
            OUR_NUMBER_OF_ROWS, NUMBER_OF_COLUMNS);
          
        }
    } else {                             // workers:
        MPI_Recv(&received,                      //  token msg received
                  1,                             //  size
                  MPI_INT,                       //  type
                  CONDUCTOR,                     //  sender
                  1,                             //  tag
                  MPI_COMM_WORLD,                //  communicator
                  &status);                      //  recv status

        mk_subgrid_copy(our_current_grid, tmp_subgrid,  OUR_NUMBER_OF_ROWS, NUMBER_OF_COLUMNS);

        MPI_Send(&tmp_subgrid,                  //  msg to send
                  (OUR_NUMBER_OF_ROWS + 2) * (NUMBER_OF_COLUMNS + 2),  //  size
                  MPI_INT,                      //  type
                  CONDUCTOR,                    //  destination
                  2,                            //  tag
                  MPI_COMM_WORLD);              //  communicator
    }
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    /* 10.4.  Determine our next grid -- 
              for each row in our subgrid, do the following: */
    for (our_current_row = 1; our_current_row <= OUR_NUMBER_OF_ROWS;
            our_current_row++)
    {
        /* 10.4.  For each column, do the following: */
        for (my_current_column = 1; my_current_column <= NUMBER_OF_COLUMNS;
                my_current_column++)
        {
            /* 10.4.1.  Initialize the count of ALIVE neighbors to 0 */
            my_number_of_alive_neighbors = 0;

            /* 10.4.2.  For each row of the cell's neighbors, do the
              *  following: */
            for (my_neighbor_row = our_current_row - 1;
                    my_neighbor_row <= our_current_row + 1;
                    my_neighbor_row++)
            {
                /* 10.4.2.1  For each column of the cell's neighbors, do
                  *  the following: */
                for (my_neighbor_column = my_current_column - 1;
                        my_neighbor_column <= my_current_column + 1;
                        my_neighbor_column++)
                {
                    /* 10.4.2.1 If the neighbor is not the cell itself,
                      *  and the neighbor is ALIVE, do the following: */
                    if ((my_neighbor_row != our_current_row
                                || my_neighbor_column != my_current_column)
                            && (our_current_grid[my_neighbor_row]
                                [my_neighbor_column] == ALIVE))
                    {
                        /* 10.4.2.1.  Add 1 to the count of the 
                          *  number of ALIVE neighbors */
                        my_number_of_alive_neighbors++;
                    }
                }
            }

            /* 10.4.3.  Apply Rule 1 of Conway's Game of Life */
            if (my_number_of_alive_neighbors < 2)
            {
                our_next_grid[our_current_row][my_current_column] = DEAD;
            }

            /* 10.4.3.  Apply Rule 2 of Conway's Game of Life */
            if (our_current_grid[our_current_row][my_current_column] == ALIVE
                    && (my_number_of_alive_neighbors == 2
                        || my_number_of_alive_neighbors == 3))
            {
                our_next_grid[our_current_row][my_current_column] = ALIVE;
            }

            /* 10.4.3.  Apply Rule 3 of Conway's Game of Life */
            if (my_number_of_alive_neighbors > 3)
            {
                our_next_grid[our_current_row][my_current_column] = DEAD;
            }

            /* 10.4.3.  Apply Rule 4 of Conway's Game of Life */
            if (our_current_grid[our_current_row][my_current_column] == DEAD
                    && my_number_of_alive_neighbors == 3)
            {
                our_next_grid[our_current_row][my_current_column] = ALIVE;
            }
        }
    }

    /* 10.5.  Copy the next subgrid into the current subgrid */
    for (our_current_row = 1; our_current_row <= OUR_NUMBER_OF_ROWS;
            our_current_row++)
    {
        for (my_current_column = 1; my_current_column <= NUMBER_OF_COLUMNS;
                my_current_column++)
        {
            our_current_grid[our_current_row][my_current_column] =
                our_next_grid[our_current_row][my_current_column];
        }
    }
  }  // END OF SIMULATION LOOP

  /* TODO  12. End the timing, using a barrier, and 
    * conductor node 0  gets the end time prints the total time.
    *  See patternlet for timing with barrier.
    * Leave the #ifndef line and #endif line and place your code inside.
    */
#ifndef SHOW_RESULTS
// Your code here
  MPI_Barrier(MPI_COMM_WORLD);
  if (OUR_RANK == CONDUCTOR) {
    totalTime = MPI_Wtime() - startTime;
    printf("%f", totalTime);
  }
#endif

  /* 11.  Deallocate data structures */
  for (our_current_row = OUR_NUMBER_OF_ROWS + 1; our_current_row >= 0;
          our_current_row--)
  {
      free(our_next_grid[our_current_row]);
      free(our_current_grid[our_current_row]);
  }
  free(our_next_grid);
  free(our_current_grid);

  /* 13.  Finalize the distributed memory environment */
  exit_if((MPI_Finalize() != MPI_SUCCESS), "MPI_Finalize", OUR_RANK);

  return 0;
}
/******* end of main ***********************/

// For debugging, copy our subgrid into contiguous memory
// to send to conductor process to print.
//
void mk_subgrid_copy(int **our_current_grid, int *grid_copy,
              int OUR_NUMBER_OF_ROWS, int NUMBER_OF_COLUMNS) {

  int our_current_row;
  int my_current_column;
      
  for(our_current_row = 0; our_current_row <= OUR_NUMBER_OF_ROWS + 1;
          our_current_row++)
  {
      for(my_current_column = 0;
              my_current_column <= NUMBER_OF_COLUMNS + 1; 
              my_current_column++)
      {
        grid_copy[our_current_row*(NUMBER_OF_COLUMNS+2) + my_current_column] = 
          our_current_grid[our_current_row][my_current_column];
      }
  }
}

/* for debugging, the conductor process prints the subgrid it receives
* from each process. This function does the pretty printing to visualize
* the ghost rows.
*/
void printGrid(int *our_current_grid, 
              int OUR_NUMBER_OF_ROWS, int NUMBER_OF_COLUMNS) { 

  int our_current_row;
  int my_current_column;
      
  for(our_current_row = 0; our_current_row <= OUR_NUMBER_OF_ROWS + 1;
          our_current_row++)
  {
    if(our_current_row == 1)
    {
        for(my_current_column = 0;
                my_current_column <= NUMBER_OF_COLUMNS + 1 + 2;
                my_current_column++)
        {
            printf("- ");
        }
        printf("\n");
    }

    for(my_current_column = 0;
            my_current_column <= NUMBER_OF_COLUMNS + 1; 
            my_current_column++)
    {
        if(my_current_column == 1)
        {
            printf("| ");
        }

        printf("%d ", our_current_grid[our_current_row*(NUMBER_OF_COLUMNS+2) + my_current_column] );

        if(my_current_column == NUMBER_OF_COLUMNS)
        {
            printf("| ");
        }
    }
    printf("\n");

    if(our_current_row == OUR_NUMBER_OF_ROWS)
    {
        for(my_current_column = 0;
                my_current_column <= NUMBER_OF_COLUMNS + 1 + 2;
                my_current_column++)
        {
            printf("- ");
        }
        printf("\n");
    }
  }

}
