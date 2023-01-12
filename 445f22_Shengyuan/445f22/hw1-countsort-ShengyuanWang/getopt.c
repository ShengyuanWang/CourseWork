/*
	example of command line parsing via getopt
	usage: getopt [-dmp] -f fname [-s sname] name [name ...]

	original author:
	Paul Krzyzanowski
	found at: https://www.cs.rutgers.edu/~pxk/416/notes/c-tutorials/getopt.html

	updated by Libby Shoop, Macalester College

*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int debug = 0;

int
main(int argc, char **argv)
{
	// these two variables with these names have become standard
	extern char *optarg;  // holds a string for text found after the 'flag'
	extern int optind;
	// c = return from each call of getopt
	// err used to indicate an error condition
	int c, err = 0; 
	int mflag=0, pflag=0, fflag=0;  //values set if -m,-p or -f used
	// these next two hold strings after -f or -s
	// note in the call to getopt in while below that f: and s:
	// indicate that these flags should have a string after them
	char *sname = "default_sname", *fname;

	// This usage string is  a kind thing to create for users so 
	// they can determine how to run the program if something goes
	// wrong.
	static char usage[] = "usage: %s [-dmp] -f fname [-s sname] name [name ...]\n";

	while ((c = getopt(argc, argv, "df:mps:")) != -1)
		switch (c) {
		case 'd':
			debug = 1;
			break;
		case 'm':
			mflag = 1;
			break;
		case 'p':
			pflag = 1;
			break;
		case 'f':
			fflag = 1;
			fname = optarg;
			break;
		case 's':
			sname = optarg;
			break;
		case '?':
			err = 1;
			break;
		}
	if (fflag == 0) {	/* -f was mandatory */
		fprintf(stderr, "%s: missing -f option\n", argv[0]);
		fprintf(stderr, usage, argv[0]);
		exit(1);
	} else if ((optind+1) > argc) {	
		/* need at least one argument (change +1 to +2 for two, etc. as needeed) */

		printf("optind = %d, argc=%d\n", optind, argc);
		fprintf(stderr, "%s: missing name\n", argv[0]);
		fprintf(stderr, usage, argv[0]);
		exit(1);
	} else if (err) {
		fprintf(stderr, usage, argv[0]);
		exit(1);
	}
	/* see what we have */
	printf("debug = %d\n", debug);
	printf("pflag = %d\n", pflag);
	printf("mflag = %d\n", mflag);
	printf("fname = \"%s\"\n", fname);
	printf("sname = \"%s\"\n", sname);
	
	if (optind < argc)	/* these are the arguments after the command-line options */
		for (; optind < argc; optind++)
			printf("argument: \"%s\"\n", argv[optind]);
	else {
		printf("no arguments left to process\n");
	}
	exit(0);
}
