#include "gnuplot_i.h"

void draw2DHeat() {
    gnuplot_ctrl * plt;
        
    plt = gnuplot_init();
    gnuplot_cmd(plt,"set terminal x11 title 'initial temps'");
    gnuplot_cmd(plt, "set title 'Initial plate temps'");
    gnuplot_cmd(plt, " plot 'initial.dat' matrix with image");

    gnuplot_ctrl * plt2;
        
    plt2 = gnuplot_init();
    gnuplot_cmd(plt2,"set terminal x11 title 'final temps'");
    gnuplot_cmd(plt2, "set title 'Final plate temps'");
    gnuplot_cmd(plt2, " plot 'final.dat' matrix with image");
}