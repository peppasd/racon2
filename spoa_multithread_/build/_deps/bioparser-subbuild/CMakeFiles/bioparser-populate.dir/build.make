# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/kevin/spoa/build/_deps/bioparser-subbuild

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kevin/spoa/build/_deps/bioparser-subbuild

# Utility rule file for bioparser-populate.

# Include any custom commands dependencies for this target.
include CMakeFiles/bioparser-populate.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/bioparser-populate.dir/progress.make

CMakeFiles/bioparser-populate: CMakeFiles/bioparser-populate-complete

CMakeFiles/bioparser-populate-complete: bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-install
CMakeFiles/bioparser-populate-complete: bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-mkdir
CMakeFiles/bioparser-populate-complete: bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-download
CMakeFiles/bioparser-populate-complete: bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-update
CMakeFiles/bioparser-populate-complete: bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-patch
CMakeFiles/bioparser-populate-complete: bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-configure
CMakeFiles/bioparser-populate-complete: bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-build
CMakeFiles/bioparser-populate-complete: bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-install
CMakeFiles/bioparser-populate-complete: bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-test
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/kevin/spoa/build/_deps/bioparser-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Completed 'bioparser-populate'"
	/usr/bin/cmake -E make_directory /home/kevin/spoa/build/_deps/bioparser-subbuild/CMakeFiles
	/usr/bin/cmake -E touch /home/kevin/spoa/build/_deps/bioparser-subbuild/CMakeFiles/bioparser-populate-complete
	/usr/bin/cmake -E touch /home/kevin/spoa/build/_deps/bioparser-subbuild/bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-done

bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-update:
.PHONY : bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-update

bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-build: bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-configure
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/kevin/spoa/build/_deps/bioparser-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "No build step for 'bioparser-populate'"
	cd /home/kevin/spoa/build/_deps/bioparser-build && /usr/bin/cmake -E echo_append
	cd /home/kevin/spoa/build/_deps/bioparser-build && /usr/bin/cmake -E touch /home/kevin/spoa/build/_deps/bioparser-subbuild/bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-build

bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-configure: bioparser-populate-prefix/tmp/bioparser-populate-cfgcmd.txt
bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-configure: bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-patch
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/kevin/spoa/build/_deps/bioparser-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "No configure step for 'bioparser-populate'"
	cd /home/kevin/spoa/build/_deps/bioparser-build && /usr/bin/cmake -E echo_append
	cd /home/kevin/spoa/build/_deps/bioparser-build && /usr/bin/cmake -E touch /home/kevin/spoa/build/_deps/bioparser-subbuild/bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-configure

bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-download: bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-gitinfo.txt
bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-download: bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-mkdir
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/kevin/spoa/build/_deps/bioparser-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Performing download step (git clone) for 'bioparser-populate'"
	cd /home/kevin/spoa/build/_deps && /usr/bin/cmake -P /home/kevin/spoa/build/_deps/bioparser-subbuild/bioparser-populate-prefix/tmp/bioparser-populate-gitclone.cmake
	cd /home/kevin/spoa/build/_deps && /usr/bin/cmake -E touch /home/kevin/spoa/build/_deps/bioparser-subbuild/bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-download

bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-install: bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-build
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/kevin/spoa/build/_deps/bioparser-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "No install step for 'bioparser-populate'"
	cd /home/kevin/spoa/build/_deps/bioparser-build && /usr/bin/cmake -E echo_append
	cd /home/kevin/spoa/build/_deps/bioparser-build && /usr/bin/cmake -E touch /home/kevin/spoa/build/_deps/bioparser-subbuild/bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-install

bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-mkdir:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/kevin/spoa/build/_deps/bioparser-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Creating directories for 'bioparser-populate'"
	/usr/bin/cmake -E make_directory /home/kevin/spoa/build/_deps/bioparser-src
	/usr/bin/cmake -E make_directory /home/kevin/spoa/build/_deps/bioparser-build
	/usr/bin/cmake -E make_directory /home/kevin/spoa/build/_deps/bioparser-subbuild/bioparser-populate-prefix
	/usr/bin/cmake -E make_directory /home/kevin/spoa/build/_deps/bioparser-subbuild/bioparser-populate-prefix/tmp
	/usr/bin/cmake -E make_directory /home/kevin/spoa/build/_deps/bioparser-subbuild/bioparser-populate-prefix/src/bioparser-populate-stamp
	/usr/bin/cmake -E make_directory /home/kevin/spoa/build/_deps/bioparser-subbuild/bioparser-populate-prefix/src
	/usr/bin/cmake -E make_directory /home/kevin/spoa/build/_deps/bioparser-subbuild/bioparser-populate-prefix/src/bioparser-populate-stamp
	/usr/bin/cmake -E touch /home/kevin/spoa/build/_deps/bioparser-subbuild/bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-mkdir

bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-patch: bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-update
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/kevin/spoa/build/_deps/bioparser-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "No patch step for 'bioparser-populate'"
	/usr/bin/cmake -E echo_append
	/usr/bin/cmake -E touch /home/kevin/spoa/build/_deps/bioparser-subbuild/bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-patch

bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-update:
.PHONY : bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-update

bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-test: bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-install
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/kevin/spoa/build/_deps/bioparser-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "No test step for 'bioparser-populate'"
	cd /home/kevin/spoa/build/_deps/bioparser-build && /usr/bin/cmake -E echo_append
	cd /home/kevin/spoa/build/_deps/bioparser-build && /usr/bin/cmake -E touch /home/kevin/spoa/build/_deps/bioparser-subbuild/bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-test

bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-update: bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-download
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/kevin/spoa/build/_deps/bioparser-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Performing update step for 'bioparser-populate'"
	cd /home/kevin/spoa/build/_deps/bioparser-src && /usr/bin/cmake -P /home/kevin/spoa/build/_deps/bioparser-subbuild/bioparser-populate-prefix/tmp/bioparser-populate-gitupdate.cmake

bioparser-populate: CMakeFiles/bioparser-populate
bioparser-populate: CMakeFiles/bioparser-populate-complete
bioparser-populate: bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-build
bioparser-populate: bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-configure
bioparser-populate: bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-download
bioparser-populate: bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-install
bioparser-populate: bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-mkdir
bioparser-populate: bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-patch
bioparser-populate: bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-test
bioparser-populate: bioparser-populate-prefix/src/bioparser-populate-stamp/bioparser-populate-update
bioparser-populate: CMakeFiles/bioparser-populate.dir/build.make
.PHONY : bioparser-populate

# Rule to build all files generated by this target.
CMakeFiles/bioparser-populate.dir/build: bioparser-populate
.PHONY : CMakeFiles/bioparser-populate.dir/build

CMakeFiles/bioparser-populate.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/bioparser-populate.dir/cmake_clean.cmake
.PHONY : CMakeFiles/bioparser-populate.dir/clean

CMakeFiles/bioparser-populate.dir/depend:
	cd /home/kevin/spoa/build/_deps/bioparser-subbuild && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kevin/spoa/build/_deps/bioparser-subbuild /home/kevin/spoa/build/_deps/bioparser-subbuild /home/kevin/spoa/build/_deps/bioparser-subbuild /home/kevin/spoa/build/_deps/bioparser-subbuild /home/kevin/spoa/build/_deps/bioparser-subbuild/CMakeFiles/bioparser-populate.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/bioparser-populate.dir/depend
