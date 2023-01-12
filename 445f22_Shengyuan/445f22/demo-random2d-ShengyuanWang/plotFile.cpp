/*****************************************************************

This contains the functions to create text files to be used with gnuplot
after the program is run.

The full simulation will create two files:
    ising_2d_initial.txt
    ising_2d_final.txt

These can then be transformed into images using gnuplot on the 
 command line like this:
      gnuplot ising_2d_initial.txt
      gnuplot ising_2d_final.txt

Executing each of these creates the following files respectively:
     ising_2d_initial.png
     ising_2d_final.png

*****************************************************************/

# include <stdio.h>

/******************************************************************************/

void plot_file ( int m, int n, int c1[], char const *title, 
                 char const *plot_filename, char const *png_filename )

/******************************************************************************/
/*
  Purpose:

    PLOT_FILE writes the current configuration to a GNUPLOT plot file.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    30 June 2013

  Author:

    John Burkardt

  Parameters:

    Input, int M, N, the number of rows and columns.

    Input, int C1[M*N], the current state of the system.

    Input, char *TITLE, a title for the plot.

    Input, char *PLOT_FILENAME, a name for the GNUPLOT
    command file to be created.

    Input, char *PNG_FILENAME, the name of the PNG graphics
    file to be created.
*/
{
  int i;
  int j;
  FILE *plot_unit;
  double ratio;
  int x1;
  int x2;
  int y1;
  int y2;

  plot_unit = fopen ( plot_filename, "wt" );

  ratio = ( double ) ( n ) / ( double ) ( m );

  fprintf ( plot_unit, "set term png\n" );
  fprintf ( plot_unit, "set output \"%s\"\n", png_filename );
  fprintf ( plot_unit, "set xrange [ 0 : %d ]\n", m );
  fprintf ( plot_unit, "set yrange [ 0 : %d ]\n", n );
  fprintf ( plot_unit, "set nokey\n" );
  fprintf ( plot_unit, "set title \"%s\"\n", title );
  fprintf ( plot_unit, "unset tics\n" );

  fprintf  ( plot_unit, "set size ratio %g\n", ratio );
  for ( j = 0; j < n; j++ )
  {
    y1 = j;
    y2 = j + 1;
    for ( i = 0; i < m; i++ )
    {
      x1 = m - i - 1;
      x2 = m - i;
      if ( c1[i+j*m] < 0 )
      {
        fprintf ( plot_unit, 
          "set object rectangle from %d, %d to %d, %d fc rgb 'blue'\n", 
          x1, y1, x2, y2 );
      }
      else
      {
        fprintf ( plot_unit, 
          "set object rectangle from %d, %d to %d, %d fc rgb 'red'\n", 
          x1, y1, x2, y2 );
      }
    }
  }

  fprintf  ( plot_unit, "plot 1\n" );
  fprintf  ( plot_unit, "quit\n" );

  fclose ( plot_unit );

  return;
}
