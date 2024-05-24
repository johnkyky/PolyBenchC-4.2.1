#!/bin/bash

build_std=/tmp/build_std
build_kokkos=/tmp/build_kokkos
output_dir=/tmp/output

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
	printf "%s\n" "-------------------------------------------------------------------------------------------------------"
}

function display_row_title() {
	printf "%-2s %-20s %-30s %-30s %-15s %-2s\n" "|" $1 "Time standard version" "Time kokkos version" "Check output" "|"
}

function display_row_data() {
	printf "%-2s %-20s %-30s %-30s $4%-15s${NC} %-2s\n" "|" $1 $2 $3 $5 "|"
}

function measure_time() {
	local start end elapsed

	start=$(date +%s%3N)
	"$@"
	end=$(date +%s%3N)

	elapsed=$((end - start))
	echo $elapsed
}

function run_polybench() {
	local kernel_dir=$1
	shift
	local kernel_list=("$@")

	display_row_line
	display_row_title $kernel_dir
	display_row_line

	for kernel in ${kernel_list[@]}; do
		cd $build_std
		echo_replace "Building ${kernel}_std"
		make -j $kernel >>$output_dir/log.txt
		check_exit_code "Error building ${kernel}_std"
		echo_replace "Running $kernel standard version"
		time_std=$(measure_time $build_std/${kernel_dir}/$kernel/$kernel 2>$output_dir/${kernel}_std.out)
		check_exit_code "Error running ${kernel}_std"

		cd $build_kokkos
		echo_replace "Building ${kernel}_kokkos"
		make -j $kernel >>$output_dir/log.txt
		check_exit_code "Error building ${kernel}_kokkos"
		echo_replace "Running $kernel kokkos version"
		time_kokkos=$(measure_time $build_kokkos/${kernel_dir}/$kernel/$kernel 2>$output_dir/${kernel}_kokkos.out)
		check_exit_code "Error running ${kernel}_kokkos"

		echo_replace "Comparing output"

		# check_output="┌∩┐(◣_◢)┌∩┐"
		check_output="X"
		check_color=$RED
		if cmp -s $output_dir/${kernel}_std.out $output_dir/${kernel}_kokkos.out; then
			check_output="V"
			check_color=$GREEN
		fi

		echo_replace ""
		display_row_data $kernel $time_std $time_kokkos $check_color $check_output
		display_row_line
	done
}

rm -fr $build_std $build_kokkos $output_dir

echo_replace "Creating build directories"
mkdir -p $build_std
mkdir -p $build_kokkos
mkdir -p $output_dir

polybench_dir=$(pwd)/..

echo_replace "Generating build files for Polybench standard versionalallalla\r"
cmake -S $polybench_dir -B $build_std -DCMAKE_BUILD_TYPE=Release -DPB_DUMP_ARRAYS=ON >>$output_dir/log.txt

echo_replace "Generating build files for Polybench Kokkos version\r"
cmake -S $polybench_dir -B $build_kokkos -DCMAKE_BUILD_TYPE=Release -DPB_DUMP_ARRAYS=ON -DPB_KOKKOS=ON -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=ON >>$output_dir/log.txt

# set variable to the benchmarks you want to running
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# datamining
dataminings_dir=datamining
dataminings_kernel=("correlation" "covariance")
# run_polybench ${dataminings_dir} "${dataminings_kernel[@]}"

echo -e "\n"

# stencils
kernel_dir=stencils
stencils_kernel=("adi" "fdtd-2d" "heat-3d" "jacobi-1d" "jacobi-2d" "seidel-2d")
run_polybench ${kernel_dir} "${stencils_kernel[@]}"

# linear-algebra
# linear-algebra/kernels/2mm/2mm.c
# linear-algebra/kernels/3mm/3mm.c
# linear-algebra/kernels/atax/atax.c
# linear-algebra/kernels/bicg/bicg.c
# linear-algebra/kernels/cholesky/cholesky.c
# linear-algebra/kernels/doitgen/doitgen.c
# linear-algebra/kernels/gemm/gemm.c
# linear-algebra/kernels/gemver/gemver.c
# linear-algebra/kernels/gesummv/gesummv.c
# linear-algebra/kernels/mvt/mvt.c
# linear-algebra/kernels/symm/symm.c
# linear-algebra/kernels/syr2k/syr2k.c
# linear-algebra/kernels/syrk/syrk.c
# linear-algebra/kernels/trisolv/trisolv.c
# linear-algebra/kernels/trmm/trmm.c

# linear-algebra/solvers/durbin/durbin.c
# linear-algebra/solvers/dynprog/dynprog.c
# linear-algebra/solvers/gramschmidt/gramschmidt.c
# linear-algebra/solvers/lu/lu.c
# linear-algebra/solvers/ludcmp/ludcmp.c

# medley
# medley/floyd-warshall/floyd-warshall.c
# medley/reg_detect/reg_detect.c
