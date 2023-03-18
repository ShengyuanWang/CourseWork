#!/bin/sh
set -e
if test "$CONFIGURATION" = "Debug"; then :
  cd /Users/shengyuanwang/Desktop/CourseWork/comp465/code/CoordinateFrame-Example/build
  make -f /Users/shengyuanwang/Desktop/CourseWork/comp465/code/CoordinateFrame-Example/build/CMakeScripts/ReRunCMake.make
fi
if test "$CONFIGURATION" = "Release"; then :
  cd /Users/shengyuanwang/Desktop/CourseWork/comp465/code/CoordinateFrame-Example/build
  make -f /Users/shengyuanwang/Desktop/CourseWork/comp465/code/CoordinateFrame-Example/build/CMakeScripts/ReRunCMake.make
fi
if test "$CONFIGURATION" = "MinSizeRel"; then :
  cd /Users/shengyuanwang/Desktop/CourseWork/comp465/code/CoordinateFrame-Example/build
  make -f /Users/shengyuanwang/Desktop/CourseWork/comp465/code/CoordinateFrame-Example/build/CMakeScripts/ReRunCMake.make
fi
if test "$CONFIGURATION" = "RelWithDebInfo"; then :
  cd /Users/shengyuanwang/Desktop/CourseWork/comp465/code/CoordinateFrame-Example/build
  make -f /Users/shengyuanwang/Desktop/CourseWork/comp465/code/CoordinateFrame-Example/build/CMakeScripts/ReRunCMake.make
fi

