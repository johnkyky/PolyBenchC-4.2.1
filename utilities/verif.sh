#!/bin/bash

### variables
process_dir=/tmp/polybench
build_std=${process_dir}/build_std
build_kokkos=${process_dir}/build_kokkos
build_polly=${process_dir}/build_polly
output_dir=${process_dir}/output

kokkos_install_dir=/home/johnkyky/Documents/Phd_project/kokkos/install/lib64/cmake/Kokkos
llvm_install_dir=/home/johnkyky/Documents/Phd_project/llvm/install/bin/clang++

polybench_dir=$(pwd)/..
dataset="MINI"

###

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[0;37m'
NC='\033[0m' # No Color

function echo_color() {
  local color="$1"
  local message="$2"
  echo -e "${color}${message}${NC}"
}

function echo_replace() {
  echo -ne "\r\033[K"
  echo -ne $1
}

function check_exit_code() {
  if [ $? -ne 0 ]; then
    echo_color $RED "Error: $1"
    exit 1
  fi
}

function display_row_line() {
  printf "%s\n" " ----------------------------------------------------------------------------------------------------------------------------------------------"
}

function display_row_title() {
  local kernel=$1
  printf "%-2s %-30s %-30s %-30s %-30s %-15s %-2s\n" "|" ${kernel} "Time standard version" "Time kokkos version" "Time polly version" "Check output" "|"
}

function display_row_data() {
  local kernel=$1
  local time_str=$2
  local time_kokkos=$3
  local time_polly=$4
  local check_color=$5
  local check_char=$6
  printf "%-2s %-30s %-30s %-30s %-30s ${check_color}%-15s${NC} %-2s\n" "|" ${kernel} ${time_str} ${time_kokkos} ${time_polly} ${check_char} "|"
}

function measure_time() {
  local start end elapsed

  start=$(date +%s%3N)
  "$@"
  end=$(date +%s%3N)

  elapsed=$((end - start))
  echo $elapsed
}

function run() {
  "$@"
  echo zizi
}

function run_polybench() {
  local kernel_dir=$1
  shift
  local kernel_list=("$@")

  display_row_line
  display_row_title $kernel_dir
  display_row_line

  for kernel in ${kernel_list[@]}; do
    mkdir -p ${output_dir}/${kernel_dir}/${kernel}

    cd ${build_std}
    echo_replace "Building ${kernel} standard version"
    make -j $kernel &>${output_dir}/${kernel_dir}/${kernel}/${kernel}_std.compile
    check_exit_code "Error building ${kernel}_std"
    echo_replace "Running ${kernel} standard version"
    ${build_std}/${kernel_dir}/${kernel}/${kernel} >${output_dir}/${kernel_dir}/${kernel}/${kernel}_std.time 2>${output_dir}/${kernel_dir}/${kernel}/${kernel}_std.out
    check_exit_code "Error running ${kernel}_std"
    time_std=$(<${output_dir}/${kernel_dir}/${kernel}/${kernel}_std.time)

    cd ${build_kokkos}
    echo_replace "Building ${kernel} kokkos version"
    make -j $kernel &>${output_dir}/${kernel_dir}/${kernel}/${kernel}_kokkos.compile
    check_exit_code "Error building ${kernel}_kokkos"
    echo_replace "Running ${kernel} kokkos version"
    ${build_kokkos}/${kernel_dir}/${kernel}/${kernel} >${output_dir}/${kernel_dir}/${kernel}/${kernel}_kokkos.time 2>${output_dir}/${kernel_dir}/${kernel}/${kernel}_kokkos.out
    check_exit_code "Error running ${kernel}_kokkos"
    time_kokkos=$(<${output_dir}/${kernel_dir}/${kernel}/${kernel}_kokkos.time)

    cd ${build_polly}
    echo_replace "Building ${kernel} polly version"
    make -j $kernel &>${output_dir}/${kernel_dir}/${kernel}/${kernel}_polly.compile
    check_exit_code "Error building ${kernel}_polly"
    echo_replace "Running ${kernel} polly version"
    ${build_polly}/${kernel_dir}/${kernel}/${kernel} >${output_dir}/${kernel_dir}/${kernel}/${kernel}_polly.time 2>${output_dir}/${kernel_dir}/${kernel}/${kernel}_polly.out
    check_exit_code "Error running ${kernel}_polly"
    time_polly=$(<${output_dir}/${kernel_dir}/${kernel}/${kernel}_polly.time)

    echo_replace "Comparing output"

    # check_output="┌∩┐(◣_◢)┌∩┐"
    check_output="V"
    check_color=$GREEN
    if ! cmp -s ${output_dir}/${kernel_dir}/${kernel}/${kernel}_std.out ${output_dir}/${kernel_dir}/${kernel}/${kernel}_kokkos.out; then
      check_output="X"
      check_color=$RED
    fi
    if ! cmp -s ${output_dir}/${kernel_dir}/${kernel}/${kernel}_std.out ${output_dir}/${kernel_dir}/${kernel}/${kernel}_polly.out; then
      check_output="X"
      check_color=$RED
    fi

    echo_replace ""
    display_row_data ${kernel} ${time_std} ${time_kokkos} ${time_polly} ${check_color} ${check_output}
    display_row_line
  done
}

rm -fr ${process_dir}

echo_replace "Creating build directories"
mkdir -p $process_dir
mkdir -p $build_std
mkdir -p $build_kokkos
mkdir -p $build_polly
mkdir -p $output_dir

echo_replace "Generating build files for Polybench standard version\r"
cmake -S $polybench_dir \
  -B $build_std \
  -DCMAKE_CXX_COMPILER="/home/johnkyky/Documents/Phd_project/llvm/install/bin/clang++" \
  -DCMAKE_BUILD_TYPE=Release \
  -DPB_CYCLE_MONITORING=ON \
  -DPB_DUMP_ARRAYS=ON \
  -DPB_DATASET_SIZE=${dataset} >>$output_dir/cmake_std.log

echo_replace "Generating build files for Polybench Kokkos version\r"
cmake -S $polybench_dir \
  -B $build_kokkos \
  -DCMAKE_CXX_COMPILER=${llvm_install_dir} \
  -DCMAKE_BUILD_TYPE=Release \
  -DPB_CYCLE_MONITORING=ON \
  -DPB_KOKKOS=ON \
  -DPB_KOKKOS_DIR=${kokkos_install_dir} \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=OFF \
  -DPB_DUMP_ARRAYS=ON \
  -DPB_DATASET_SIZE=${dataset} >>$output_dir/cmake_kokkos.log

echo_replace "Generating build files for Polybench Kokkos version with polly\r"
cmake -S $polybench_dir \
  -B $build_polly \
  -DCMAKE_CXX_COMPILER=${llvm_install_dir} \
  -DCMAKE_BUILD_TYPE=Release \
  -DPB_CYCLE_MONITORING=ON \
  -DPB_KOKKOS=ON \
  -DPB_KOKKOS_DIR=${kokkos_install_dir} \
  -DPB_USE_POLLY=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DPB_DUMP_ARRAYS=ON \
  -DPB_DATASET_SIZE=${dataset} >>$output_dir/cmake_polly.log

# set variable to the benchmarks you want to running
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# datamining
dataminings_dir=datamining
dataminings_kernel=("correlation" "covariance")
run_polybench ${dataminings_dir} "${dataminings_kernel[@]}"

echo -e "\n"

# linear-algebra kernels
kernel_dir=linear-algebra/kernels
linear_algebra_kernel=("2mm" "3mm" "atax" "bicg" "doitgen" "mvt")
run_polybench ${kernel_dir} "${linear_algebra_kernel[@]}"

# linear-algebra solvers
kernel_dir=linear-algebra/solvers
linear_algebra_kernel=("cholesky" "durbin" "gramschmidt" "lu" "ludcmp" "trisolv")
run_polybench ${kernel_dir} "${linear_algebra_kernel[@]}"

echo -e "\n"

# medley
kernel_dir=medley
medley_kernel=("deriche" "floyd-warshall" "nussinov")
run_polybench ${kernel_dir} "${medley_kernel[@]}"

echo -e "\n"

# stencils
kernel_dir=stencils
stencils_kernel=("adi" "fdtd-2d" "heat-3d" "jacobi-1d" "jacobi-2d" "seidel-2d")
run_polybench ${kernel_dir} "${stencils_kernel[@]}"

# linear-algebra/blas
