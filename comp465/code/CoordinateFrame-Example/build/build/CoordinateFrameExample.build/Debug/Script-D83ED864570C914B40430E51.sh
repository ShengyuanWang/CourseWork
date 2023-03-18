#!/bin/sh
set -e
if test "$CONFIGURATION" = "Debug"; then :
  cd /Users/shengyuanwang/Desktop/CourseWork/comp465/code/CoordinateFrame-Example/build
  /Applications/CMake.app/Contents/bin/cmake -E copy_directory /Users/shengyuanwang/Desktop/CourseWork/comp465/code/basicgraphics/resources /Users/shengyuanwang/Desktop/CourseWork/comp465/code/CoordinateFrame-Example/build/Debug/
fi
if test "$CONFIGURATION" = "Release"; then :
  cd /Users/shengyuanwang/Desktop/CourseWork/comp465/code/CoordinateFrame-Example/build
  /Applications/CMake.app/Contents/bin/cmake -E copy_directory /Users/shengyuanwang/Desktop/CourseWork/comp465/code/basicgraphics/resources /Users/shengyuanwang/Desktop/CourseWork/comp465/code/CoordinateFrame-Example/build/Release/
fi
if test "$CONFIGURATION" = "MinSizeRel"; then :
  cd /Users/shengyuanwang/Desktop/CourseWork/comp465/code/CoordinateFrame-Example/build
  /Applications/CMake.app/Contents/bin/cmake -E copy_directory /Users/shengyuanwang/Desktop/CourseWork/comp465/code/basicgraphics/resources /Users/shengyuanwang/Desktop/CourseWork/comp465/code/CoordinateFrame-Example/build/MinSizeRel/
fi
if test "$CONFIGURATION" = "RelWithDebInfo"; then :
  cd /Users/shengyuanwang/Desktop/CourseWork/comp465/code/CoordinateFrame-Example/build
  /Applications/CMake.app/Contents/bin/cmake -E copy_directory /Users/shengyuanwang/Desktop/CourseWork/comp465/code/basicgraphics/resources /Users/shengyuanwang/Desktop/CourseWork/comp465/code/CoordinateFrame-Example/build/RelWithDebInfo/
fi

