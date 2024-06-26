# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.29

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.29.3/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.29.3/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/michaeltamburello/CLionProjects/cortex

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/michaeltamburello/CLionProjects/cortex/build

# Include any dependencies generated for this target.
include CMakeFiles/cortex_layers.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/cortex_layers.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/cortex_layers.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cortex_layers.dir/flags.make

CMakeFiles/cortex_layers.dir/src/layers/linear.cpp.o: CMakeFiles/cortex_layers.dir/flags.make
CMakeFiles/cortex_layers.dir/src/layers/linear.cpp.o: /Users/michaeltamburello/CLionProjects/cortex/src/layers/linear.cpp
CMakeFiles/cortex_layers.dir/src/layers/linear.cpp.o: CMakeFiles/cortex_layers.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/michaeltamburello/CLionProjects/cortex/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/cortex_layers.dir/src/layers/linear.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/cortex_layers.dir/src/layers/linear.cpp.o -MF CMakeFiles/cortex_layers.dir/src/layers/linear.cpp.o.d -o CMakeFiles/cortex_layers.dir/src/layers/linear.cpp.o -c /Users/michaeltamburello/CLionProjects/cortex/src/layers/linear.cpp

CMakeFiles/cortex_layers.dir/src/layers/linear.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/cortex_layers.dir/src/layers/linear.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/michaeltamburello/CLionProjects/cortex/src/layers/linear.cpp > CMakeFiles/cortex_layers.dir/src/layers/linear.cpp.i

CMakeFiles/cortex_layers.dir/src/layers/linear.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/cortex_layers.dir/src/layers/linear.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/michaeltamburello/CLionProjects/cortex/src/layers/linear.cpp -o CMakeFiles/cortex_layers.dir/src/layers/linear.cpp.s

CMakeFiles/cortex_layers.dir/src/activations/relu.cpp.o: CMakeFiles/cortex_layers.dir/flags.make
CMakeFiles/cortex_layers.dir/src/activations/relu.cpp.o: /Users/michaeltamburello/CLionProjects/cortex/src/activations/relu.cpp
CMakeFiles/cortex_layers.dir/src/activations/relu.cpp.o: CMakeFiles/cortex_layers.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/michaeltamburello/CLionProjects/cortex/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/cortex_layers.dir/src/activations/relu.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/cortex_layers.dir/src/activations/relu.cpp.o -MF CMakeFiles/cortex_layers.dir/src/activations/relu.cpp.o.d -o CMakeFiles/cortex_layers.dir/src/activations/relu.cpp.o -c /Users/michaeltamburello/CLionProjects/cortex/src/activations/relu.cpp

CMakeFiles/cortex_layers.dir/src/activations/relu.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/cortex_layers.dir/src/activations/relu.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/michaeltamburello/CLionProjects/cortex/src/activations/relu.cpp > CMakeFiles/cortex_layers.dir/src/activations/relu.cpp.i

CMakeFiles/cortex_layers.dir/src/activations/relu.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/cortex_layers.dir/src/activations/relu.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/michaeltamburello/CLionProjects/cortex/src/activations/relu.cpp -o CMakeFiles/cortex_layers.dir/src/activations/relu.cpp.s

CMakeFiles/cortex_layers.dir/src/activations/softmax.cpp.o: CMakeFiles/cortex_layers.dir/flags.make
CMakeFiles/cortex_layers.dir/src/activations/softmax.cpp.o: /Users/michaeltamburello/CLionProjects/cortex/src/activations/softmax.cpp
CMakeFiles/cortex_layers.dir/src/activations/softmax.cpp.o: CMakeFiles/cortex_layers.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/michaeltamburello/CLionProjects/cortex/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/cortex_layers.dir/src/activations/softmax.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/cortex_layers.dir/src/activations/softmax.cpp.o -MF CMakeFiles/cortex_layers.dir/src/activations/softmax.cpp.o.d -o CMakeFiles/cortex_layers.dir/src/activations/softmax.cpp.o -c /Users/michaeltamburello/CLionProjects/cortex/src/activations/softmax.cpp

CMakeFiles/cortex_layers.dir/src/activations/softmax.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/cortex_layers.dir/src/activations/softmax.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/michaeltamburello/CLionProjects/cortex/src/activations/softmax.cpp > CMakeFiles/cortex_layers.dir/src/activations/softmax.cpp.i

CMakeFiles/cortex_layers.dir/src/activations/softmax.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/cortex_layers.dir/src/activations/softmax.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/michaeltamburello/CLionProjects/cortex/src/activations/softmax.cpp -o CMakeFiles/cortex_layers.dir/src/activations/softmax.cpp.s

CMakeFiles/cortex_layers.dir/src/matrix.cpp.o: CMakeFiles/cortex_layers.dir/flags.make
CMakeFiles/cortex_layers.dir/src/matrix.cpp.o: /Users/michaeltamburello/CLionProjects/cortex/src/matrix.cpp
CMakeFiles/cortex_layers.dir/src/matrix.cpp.o: CMakeFiles/cortex_layers.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/michaeltamburello/CLionProjects/cortex/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/cortex_layers.dir/src/matrix.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/cortex_layers.dir/src/matrix.cpp.o -MF CMakeFiles/cortex_layers.dir/src/matrix.cpp.o.d -o CMakeFiles/cortex_layers.dir/src/matrix.cpp.o -c /Users/michaeltamburello/CLionProjects/cortex/src/matrix.cpp

CMakeFiles/cortex_layers.dir/src/matrix.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/cortex_layers.dir/src/matrix.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/michaeltamburello/CLionProjects/cortex/src/matrix.cpp > CMakeFiles/cortex_layers.dir/src/matrix.cpp.i

CMakeFiles/cortex_layers.dir/src/matrix.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/cortex_layers.dir/src/matrix.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/michaeltamburello/CLionProjects/cortex/src/matrix.cpp -o CMakeFiles/cortex_layers.dir/src/matrix.cpp.s

CMakeFiles/cortex_layers.dir/src/parameter.cpp.o: CMakeFiles/cortex_layers.dir/flags.make
CMakeFiles/cortex_layers.dir/src/parameter.cpp.o: /Users/michaeltamburello/CLionProjects/cortex/src/parameter.cpp
CMakeFiles/cortex_layers.dir/src/parameter.cpp.o: CMakeFiles/cortex_layers.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/michaeltamburello/CLionProjects/cortex/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/cortex_layers.dir/src/parameter.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/cortex_layers.dir/src/parameter.cpp.o -MF CMakeFiles/cortex_layers.dir/src/parameter.cpp.o.d -o CMakeFiles/cortex_layers.dir/src/parameter.cpp.o -c /Users/michaeltamburello/CLionProjects/cortex/src/parameter.cpp

CMakeFiles/cortex_layers.dir/src/parameter.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/cortex_layers.dir/src/parameter.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/michaeltamburello/CLionProjects/cortex/src/parameter.cpp > CMakeFiles/cortex_layers.dir/src/parameter.cpp.i

CMakeFiles/cortex_layers.dir/src/parameter.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/cortex_layers.dir/src/parameter.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/michaeltamburello/CLionProjects/cortex/src/parameter.cpp -o CMakeFiles/cortex_layers.dir/src/parameter.cpp.s

CMakeFiles/cortex_layers.dir/src/one_hot_encoder.cpp.o: CMakeFiles/cortex_layers.dir/flags.make
CMakeFiles/cortex_layers.dir/src/one_hot_encoder.cpp.o: /Users/michaeltamburello/CLionProjects/cortex/src/one_hot_encoder.cpp
CMakeFiles/cortex_layers.dir/src/one_hot_encoder.cpp.o: CMakeFiles/cortex_layers.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/michaeltamburello/CLionProjects/cortex/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/cortex_layers.dir/src/one_hot_encoder.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/cortex_layers.dir/src/one_hot_encoder.cpp.o -MF CMakeFiles/cortex_layers.dir/src/one_hot_encoder.cpp.o.d -o CMakeFiles/cortex_layers.dir/src/one_hot_encoder.cpp.o -c /Users/michaeltamburello/CLionProjects/cortex/src/one_hot_encoder.cpp

CMakeFiles/cortex_layers.dir/src/one_hot_encoder.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/cortex_layers.dir/src/one_hot_encoder.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/michaeltamburello/CLionProjects/cortex/src/one_hot_encoder.cpp > CMakeFiles/cortex_layers.dir/src/one_hot_encoder.cpp.i

CMakeFiles/cortex_layers.dir/src/one_hot_encoder.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/cortex_layers.dir/src/one_hot_encoder.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/michaeltamburello/CLionProjects/cortex/src/one_hot_encoder.cpp -o CMakeFiles/cortex_layers.dir/src/one_hot_encoder.cpp.s

CMakeFiles/cortex_layers.dir/src/dataset.cpp.o: CMakeFiles/cortex_layers.dir/flags.make
CMakeFiles/cortex_layers.dir/src/dataset.cpp.o: /Users/michaeltamburello/CLionProjects/cortex/src/dataset.cpp
CMakeFiles/cortex_layers.dir/src/dataset.cpp.o: CMakeFiles/cortex_layers.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/michaeltamburello/CLionProjects/cortex/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/cortex_layers.dir/src/dataset.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/cortex_layers.dir/src/dataset.cpp.o -MF CMakeFiles/cortex_layers.dir/src/dataset.cpp.o.d -o CMakeFiles/cortex_layers.dir/src/dataset.cpp.o -c /Users/michaeltamburello/CLionProjects/cortex/src/dataset.cpp

CMakeFiles/cortex_layers.dir/src/dataset.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/cortex_layers.dir/src/dataset.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/michaeltamburello/CLionProjects/cortex/src/dataset.cpp > CMakeFiles/cortex_layers.dir/src/dataset.cpp.i

CMakeFiles/cortex_layers.dir/src/dataset.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/cortex_layers.dir/src/dataset.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/michaeltamburello/CLionProjects/cortex/src/dataset.cpp -o CMakeFiles/cortex_layers.dir/src/dataset.cpp.s

# Object files for target cortex_layers
cortex_layers_OBJECTS = \
"CMakeFiles/cortex_layers.dir/src/layers/linear.cpp.o" \
"CMakeFiles/cortex_layers.dir/src/activations/relu.cpp.o" \
"CMakeFiles/cortex_layers.dir/src/activations/softmax.cpp.o" \
"CMakeFiles/cortex_layers.dir/src/matrix.cpp.o" \
"CMakeFiles/cortex_layers.dir/src/parameter.cpp.o" \
"CMakeFiles/cortex_layers.dir/src/one_hot_encoder.cpp.o" \
"CMakeFiles/cortex_layers.dir/src/dataset.cpp.o"

# External object files for target cortex_layers
cortex_layers_EXTERNAL_OBJECTS =

libcortex_layers.a: CMakeFiles/cortex_layers.dir/src/layers/linear.cpp.o
libcortex_layers.a: CMakeFiles/cortex_layers.dir/src/activations/relu.cpp.o
libcortex_layers.a: CMakeFiles/cortex_layers.dir/src/activations/softmax.cpp.o
libcortex_layers.a: CMakeFiles/cortex_layers.dir/src/matrix.cpp.o
libcortex_layers.a: CMakeFiles/cortex_layers.dir/src/parameter.cpp.o
libcortex_layers.a: CMakeFiles/cortex_layers.dir/src/one_hot_encoder.cpp.o
libcortex_layers.a: CMakeFiles/cortex_layers.dir/src/dataset.cpp.o
libcortex_layers.a: CMakeFiles/cortex_layers.dir/build.make
libcortex_layers.a: CMakeFiles/cortex_layers.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/michaeltamburello/CLionProjects/cortex/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Linking CXX static library libcortex_layers.a"
	$(CMAKE_COMMAND) -P CMakeFiles/cortex_layers.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cortex_layers.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cortex_layers.dir/build: libcortex_layers.a
.PHONY : CMakeFiles/cortex_layers.dir/build

CMakeFiles/cortex_layers.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cortex_layers.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cortex_layers.dir/clean

CMakeFiles/cortex_layers.dir/depend:
	cd /Users/michaeltamburello/CLionProjects/cortex/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/michaeltamburello/CLionProjects/cortex /Users/michaeltamburello/CLionProjects/cortex /Users/michaeltamburello/CLionProjects/cortex/build /Users/michaeltamburello/CLionProjects/cortex/build /Users/michaeltamburello/CLionProjects/cortex/build/CMakeFiles/cortex_layers.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/cortex_layers.dir/depend

