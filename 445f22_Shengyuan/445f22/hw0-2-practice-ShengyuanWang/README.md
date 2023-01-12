# COMP 445 HW 0 : First look at threaded programming with OpenMP

In VS Code, right-click on this file in the 'Explorer' and choose to 'Open Preview'.

## Goals

This is a simple course warm-up exercise designed to:

- Let you practice cloning an assignment from GitHub onto your laptop.
- Let you practice synchronizing it to the mscs1 server.
- Let you practice reviewing code examples and using make to compile.

- Let you practice the process for 'turning in' your assignments to GitHub. You do this by "pushing", or "syncing" your code with your GitHub repository.

- Let you see some helpful coding techniques that you will want to use throughout the course.

As such, this is a low-stakes exercise without **"points"**: please complete it by making the suggested changes to the code and making certain that those changes are on your GitHub repo.

## Now that you have this code

You should have already followed some instructions for cloning this repository in the VS Code editor and syncing a copy of it to the server. Otherwise you are reading this from GitHub and should get this repo into VS Code.

**If you haven't opened the terminal on the server, you should now. Use the 'cd' command in the command line to be in the directory where this code is.** Check with your instructor if you need help with the terminal and getting to the correct directory.

This example comes from a [long-standing tutorial about OpenMP](https://hpc-tutorials.llnl.gov/openmp/) kept at Lawrence Livermore National Labs that is is a very useful starting point for studying OpenMP. Follow the link and have a look, especially at the Introduction and OpenMP programming model. As you can see if you peruse this information, there is a lot built into OpenMP. Note from the Introduction how long this programming model has been around: parallelism has a long history in computing!

----

## <img src="img/diversity.png" alt="alt text" title="image Title" height="50"/> DEI Alert

With that long history comes a very poor nomenclature choice. It is sad to realize now that when this programming system was developed it seemed perfectly reasonable to use the term *master* to refer to the main, or primary thread of execution in a program. Unfortunately, the term still persists in the [OpenMP specification](https://www.openmp.org/spec-html/5.0/openmp.html) keywords and in many examples, including the above tutorial. Hopefully with persistence we can get this word out of the computing lexicon.

[Tech Confronts Its Use of the Labels ‘Master’ and ‘Slave’](https://www.wired.com/story/tech-confronts-use-labels-master-slave/)

[‘Master,’ ‘Slave’ and the Fight Over Offensive Terms in Computing](https://www.nytimes.com/2021/04/13/technology/racist-computer-engineering-terms-ietf.html)

[Apple to Remove 'Master/Slave' and 'Blacklist' Terms From Coding Platforms](https://www.pcmag.com/news/apple-to-remove-masterslave-and-blacklist-terms-from-coding-platforms)

[Words Matter: Finally, Tech Looks at Removing Exclusionary Language](https://thenewstack.io/words-matter-finally-tech-looks-at-removing-exclusionary-language/)

[Engineering Student Persuades Book Publisher to Remove ‘Master and Slave’ Language](https://alltogether.swe.org/2020/08/removing-master-and-slave-language/)

--------

## Some tasks for practice

### First, compile and run as is

First go to your terminal in the directory where the code is and type:

    make

Note that the makefile creates a file called *hello*. Run it like this:

    ./hello

Note that you are not yet using parallelism.

### Next, manually change number of threads in code
Now open the file hello.c in the VSCode editor. Note there are two places in the file with comments marked **TODO**. Do number 1 first.  **Pro Tip**: On Mac, cmd-/ will toggle commenting and uncommenting of a line. On PC, it is ctrl-/.

Go back to the terminal and run make again and execute the code again. Save, make, and run each time you change the nthreads value.

### Next, do it the right way: get thread number to use from command line

**Comment that code for TODO 1 back out**, and work on TODO 2. Here you may want to study the code as instructed in the TODO 2, and do a bit of research about the following:

- how to handle command-line arguments using the getopt function.

- some functions from the C stdlib library that convert strings to other data types. 

Complete the code change and build it in the terminal:

    make

Try running the code in the following ways and match the output to what is written in the code (especially the getCommandLine.c file):

    ./hello -t 4
    ./hello -t 8
    ./hello -t 4 -v
    ./hello
    ./hello -v
    ./hello -h

This last way of running the program is a standard that you should get used to doing: have a 'help' flag that when used on the command line shows how a program can be run.

Here are some error cases you should try and once again match to the code in getCommandLine.c:

    ./hello -t
    ./hello -t n

At this point it is also worth studying the Makefile if you have not already. As you move forward with more activities and projects in this course, recall how we were able to compile a utility file like getCommandLine.c into a .o file and then build it into the final executable program.

### Eventually save your work back to GitHub

At some point before class ends, work on the following instructions below with some version of the code that is working, even if it is the TODO 1 task instead of the alternative TODO 2 task.

______________________

# Keep versions and Turn in assignments by saving changes to GitHub

**IMPORTANT:** You should periodically do what we are going to describe here as you work on your homework. When you are at a point where code is working, it is always a good idea to *commit* and *push* it up to GitHub, because:

- You will then have a version that you can always get back to when something goes wrong as you continue editing.

- If you need help from an instructor over email, they can see your code on GitHub and help.

- If your laptop dies, there is a version on GitHub that you can recover.

## From VS Code editor

There are three steps to getting your code up to your GitHub repository:

1. Stage 
2. Commit
3. Push committed changes to GitHub

### 1. Stage

Open the Git panel, which should appear on the left, just below search and above debug. The changed files and new files will be shown in the top section.

In the row labeled 'Changes', hover over it and choose the + button/icon.

The section will now be called 'Staged Changes' and the files will be in it.

### 2. Commit 

In the area that says "Message", type some words to help you remember the changes you made, such as 'command line num threads.'

**Important:** A meaningful text is really helpful here, so that you can get back to this change if you need to later.

Now click the button in the line above labeled SOURCE CONTROL that looks like a check mark. The staged changes should disappear. **But you are not done yet!!**

### 2. Push committed Changes to GitHub

In the upper line labeled SOURCE CONTROL, towards the right there are 3 dots for a drop down menu. Choose to 'Push'. You should see the changes being sent in the lower left corner if you look quickly.

New versions of VS Code have added a blue button that says 'Sync Changes'. This will do the push, along with checking to see if there were changes already pushed on GitHub that were not on your copy (by doing a 'fetch' first).

You should be able to go back to GitHub and check whether your changes are now there for this repository.
