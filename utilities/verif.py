import os
import time
import shutil
import argparse
import subprocess
import hashlib
import statistics

RED = "red"
GREEN = "green"
YELLOW = "yellow"
BLUE = "blue"
MAGENTA = "magenta"
CYAN = "cyan"
WHITE = "white"
NO_COLOR = "nc"

COLOR = {
    "red": "\033[0;31m",
    "green": "\033[0;32m",
    "yellow": "\033[0;33m",
    "blue": "\033[0;34m",
    "magenta": "\033[0;35m",
    "cyan": "\033[0;36m",
    "white": "\033[0;37m",
    "nc": "\033[0m",
}

ARGS_ENV = "OMP_PROC_BIND=spread OMP_PLACES=threads "


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verif", type=int, default=1,
                        help="Run verification (1) or benchmarking (0)")
    parser.add_argument("--nb_iteration", type=int, default=5,
                        help="Number of iterations for benchmarking ignoring "
                        "for verification")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset size")
    parser.add_argument("--cxx_compiler", type=str, required=True,
                        help="C++ compiler")
    parser.add_argument("--kokkos_install_dir", type=str, required=True,
                        help="Install directory for Kokkos")
    parser.add_argument("--polybench_dir", type=str, required=True,
                        help="Polybench directory")
    args = parser.parse_args()
    return args


def display_row_line(verif):
    if verif:
        print("\r\033[K+" + "-" * 27 + "+" + "-" * 27 +
              "+" + "-" * 27 + "+" + "-" * 27 + "-" * 17 + "+")
    else:
        print("\r\033[K+" + "-" * 27 + "+" + "-" * 27 +
              "+" + "-" * 27 + "+")


def display_row_title(verif, kernel):
    display_row_line(verif)
    if verif:
        print((f"| {kernel.center(25)} | {"Standard".center(25)} | {
              "Kokkos".center(25)} | {"Polly".center(25)} |"))
    else:
        print((f"| {kernel.center(25)} | {"Kokkos".center(25)} | {
              "Polly".center(25)} |"))
    display_row_line(verif)


def display_row_data(verif,
                     kernel,
                     time_std,
                     time_kokkos,
                     time_polly,
                     check_str):
    if verif:
        print((f"| {kernel.center(25)} | {str(time_std).center(25)} | {
              str(time_kokkos).center(25)} | {str(time_polly).center(25)} | {
            check_str.center(25)} |"))
        display_row_line(verif)
    else:
        print((f"| {str(kernel).center(25)} | {str(time_kokkos).center(25)} | {
              str(time_polly).center(25)} |"))
        display_row_line(verif)


def display_row_data_bench(kernel,
                           statistics_kokkos,
                           statistics_polly):
    avg_k, med_k, std_dev_k, var_k, min_k, max_k = statistics_kokkos
    avg_p, med_p, std_dev_p, var_p, min_p, max_p = statistics_polly
    print((f"| {COLOR[GREEN]}{str(kernel).center(25)}{COLOR[NO_COLOR]} | {
        "".center(25)} | {"".center(25)} |"))
    print((f"| {"average".center(25)} | {str(f"{avg_k:,.1f}").center(25)} | {
          str(f"{avg_p:,.1f}").center(25)} |"))
    print((f"| {"median".center(25)} | {str(f"{med_k:,.1f}").center(25)} | {
          str(f"{med_p:,.1f}").center(25)} |"))
    print((f"| {"standard deviation".center(25)} | {str(f"{
        std_dev_k:,.1f}").center(25)} | {
        str(f"{std_dev_p:,.1f}").center(25)} |"))
    print((f"| {"variance".center(25)} | {str(f"{var_k:,.1f}").center(25)} | {
          str(f"{var_p:,.1f}").center(25)} |"))
    print((f"| {"minimum".center(25)} | {str(f"{min_k:,.1f}").center(25)} | {
          str(f"{min_p:,.1f}").center(25)} |"))
    print((f"| {"max".center(25)} | {str(f"{max_k:,.1f}").center(25)} | {
          str(f"{max_p:,.1f}").center(25)} |"))
    display_row_line(False)


def run_command(command, stdout_file=None, stderr_file=None):
    if stdout_file and not stderr_file:
        with open(stdout_file, "a") if stdout_file else subprocess.DEVNULL as log:
            result = subprocess.run(command, shell=True,
                                    stdout=log, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                print(f"Erreur lors de l'exécution de la commande: {command}")
                exit(1)
    elif stderr_file and not stdout_file:
        with open(stderr_file, "a") if stderr_file else subprocess.DEVNULL as log:
            result = subprocess.run(command, shell=True,
                                    stdout=subprocess.DEVNULL, stderr=log)
            if result.returncode != 0:
                print(f"Erreur lors de l'exécution de la commande: {command}")
                exit(1)
    elif stderr_file and stdout_file:
        with open(stdout_file, "a") as log_stdout, open(stderr_file, "a") as log_stderr:
            result = subprocess.run(command, shell=True,
                                    stdout=log_stdout, stderr=log_stderr)
            if result.returncode != 0:
                print(f"Erreur lors de l'exécution de la commande: {command}")
                exit(1)
    else:
        result = subprocess.run(command, shell=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(f"Erreur lors de l'exécution de la commande: {command}")
            exit(1)


def do_statistics(file):
    with open(file, "r") as fichier:
        values = [int(ligne.strip()) for ligne in fichier]
    average = statistics.mean(values)
    median = statistics.median(values)
    standard_deviation = statistics.stdev(values)
    variance = statistics.variance(values)
    minimum = min(values)
    maximum = max(values)
    return (average, median, standard_deviation, variance, minimum, maximum)


def compute_hash(fichier):
    hasher = hashlib.sha256()
    with open(fichier, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def check_output(file_std, file_kokkos, file_polly):
    # Calcul du hash des 3 fichiers
    hash1 = compute_hash(file_std)
    hash2 = compute_hash(file_kokkos)
    hash3 = compute_hash(file_polly)

    res = ""
    if hash1 != hash2:
        # print("Erreur : les fichiers standard et kokkos sont différents.")
        res += f"{RED}K\u0336{NO_COLOR}\n"
    if hash1 != hash3:
        # print("Erreur : les fichiers standard et polly sont différents.")
        res += f"{COLOR[RED]}P\u0336{COLOR[NO_COLOR]}"
    elif hash1 == hash2 and hash1 == hash3:
        # print("Les fichiers sont identiques.")
        res = f"{COLOR[GREEN]}KP{COLOR[NO_COLOR]}"
    return res


def generate_build_file(polybench_dir,
                        output_dir,
                        build_std,
                        build_kokkos,
                        build_polly,
                        cxx_compiler,
                        kokkos_install_dir,
                        dataset,
                        verif):
    print_output = "ON" if verif else "OFF"

    cmake_command_base = (
        f"cmake -S {polybench_dir} "
        f"-DCMAKE_CXX_COMPILER={cxx_compiler} "
        "-DCMAKE_BUILD_TYPE=Release "
        "-DPB_CYCLE_MONITORING=ON "
        f"-DPB_DUMP_ARRAYS={print_output} "
        f"-DPB_DATASET_SIZE={dataset} ")

    # build standard version
    if verif:
        print(f"{COLOR[GREEN]}Building standard version{
              COLOR[NO_COLOR]}\r", end="")
        cmake_command_standard = cmake_command_base + f"-B {build_std}"
        run_command(cmake_command_standard, os.path.join(
            output_dir, "cmake_standard.log"))

    # build kokkos version
    print(f"\r\033[K\r{COLOR[GREEN]}Building Kokkos version{
          COLOR[NO_COLOR]}", end="")
    cmake_command_kokkos = cmake_command_base + (f"-B {
        build_kokkos} -DPB_KOKKOS=ON "
        f"-DPB_KOKKOS_DIR={
        kokkos_install_dir} "
        "-DKokkos_ENABLE_SERIAL=ON "
        "-DKokkos_ENABLE_OPENMP=ON")
    run_command(cmake_command_kokkos, os.path.join(
        output_dir, "cmake_kokkos.log"))

    # build polly version
    print(f"\r\033[K\r{COLOR[GREEN]}Building Polly version{
          COLOR[NO_COLOR]}\r", end="")
    cmake_command_polly = cmake_command_base + (f"-B {build_polly} "
                                                "-DPB_KOKKOS=ON "
                                                f"-DPB_KOKKOS_DIR={
                                                    kokkos_install_dir} "
                                                "-DPB_USE_POLLY=ON "
                                                "-DKokkos_ENABLE_SERIAL=ON")
    run_command(cmake_command_polly, os.path.join(
        output_dir, "cmake_polly.log"))


def run_verif(kernel_dir,
              kernels,
              output_dir,
              build_std,
              build_kokkos,
              build_polly):
    for kernel in kernels:
        kernel_output_path = f"{output_dir}/{kernel_dir}/{kernel}"
        os.makedirs(kernel_output_path, exist_ok=True)
        for build, version in [(build_std, "std"),
                               (build_kokkos, "kokkos"),
                               (build_polly, "polly")]:
            os.chdir(build)
            print(f"{COLOR[YELLOW]}Building {kernel} {
                version} version{COLOR[NO_COLOR]}\r", end="")
            make_command = f"make -j {kernel}"
            run_command(make_command, os.path.join(
                kernel_output_path, f"{kernel}_{version}.compile"))
            print(f"{COLOR[YELLOW]}\rRunning {kernel} {
                version} version{COLOR[NO_COLOR]}\r", end="")
            exec_command = f"{ARGS_ENV} {build}/{kernel_dir}/{kernel}/{kernel}"
            # time.sleep(0.3)
            run_command(exec_command,
                        os.path.join(kernel_output_path,
                                     f"{kernel}_{version}.time"),
                        os.path.join(kernel_output_path,
                                     f"{kernel}_{version}.out"))

        check_str = check_output(os.path.join(kernel_output_path,
                                              f"{kernel}_{version}.out"),
                                 os.path.join(kernel_output_path,
                                              f"{kernel}_{version}.out"),
                                 os.path.join(kernel_output_path,
                                              f"{kernel}_{version}.out"))
        display_row_data(True, kernel, 1, 1, 1, check_str)


def run_bench(kernel_dir,
              kernels,
              output_dir,
              build_kokkos,
              build_polly,
              nb_iteration):
    for kernel in kernels:
        kernel_output_path = f"{output_dir}/{kernel_dir}/{kernel}"
        os.makedirs(kernel_output_path, exist_ok=True)
        statistics = []
        for build, version in [(build_kokkos, "kokkos"),
                               (build_polly, "polly")]:
            os.chdir(build)
            print(f"\r\033[K{COLOR[YELLOW]}Building {kernel} {
                  version} version{COLOR[NO_COLOR]}", end="")
            make_command = f"make -j {kernel}"
            run_command(make_command, os.path.join(
                kernel_output_path, f"{kernel}_{version}.compile"))
            for i in range(nb_iteration):
                print(f"\r\033[K{COLOR[YELLOW]}Running {kernel} {
                    version} version (iteration {i+1}/{
                    nb_iteration}){COLOR[NO_COLOR]}", end="")
                exec_command = f"{ARGS_ENV} {
                    build}/{kernel_dir}/{kernel}/{kernel}"

                time_file = os.path.join(kernel_output_path,
                                         f"{kernel}_{version}.time")
                run_command(exec_command, time_file)
            statistics.append(do_statistics(time_file))
        print("\r\033[K", end="")
        display_row_data_bench(kernel, statistics[0], statistics[1])


def main():
    args = parse_args()

    polybench_dir = args.polybench_dir
    process_dir = "/tmp/polybench"
    build_std = os.path.join(process_dir, "build_std")
    build_kokkos = os.path.join(process_dir, "build_kokkos")
    build_polly = os.path.join(process_dir, "build_polly")
    output_dir = os.path.join(process_dir, "output")

    print(f"Run {f"benchmark {args.nb_iteration} iterations" if not args.verif
                 else "verif"}\nCompiler : {args.cxx_compiler}\nKokkos : {
        args.kokkos_install_dir}\nDataset : {args.dataset}\nOutpu directory : {
        process_dir}")

    if os.path.exists(process_dir):
        shutil.rmtree(process_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(build_std, exist_ok=True)
    os.makedirs(build_kokkos, exist_ok=True)
    os.makedirs(build_polly, exist_ok=True)

    datasets = {
        # "datamining": ["correlation", "covariance"],
        # "linear-algebra/kernels": ["2mm", "3mm", "atax", "bicg", "doitgen",
        #                            "mvt"],
        # "linear-algebra/solvers": ["cholesky", "durbin", "gramschmidt", "lu",
        #                            "ludcmp", "trisolv"],
        # "medley": ["deriche", "floyd-warshall", "nussinov"],
        "stencils": ["adi", "fdtd-2d", "heat-3d", "jacobi-1d", "jacobi-2d",
                     "seidel-2d"],
    }

    generate_build_file(polybench_dir, output_dir,
                        build_std, build_kokkos, build_polly,
                        args.cxx_compiler, args.kokkos_install_dir,
                        args.dataset, args.verif)

    for kernel_dir, kernels in datasets.items():
        display_row_title(args.verif, kernel_dir)
        if args.verif:
            run_verif(kernel_dir, kernels, output_dir,
                      build_std, build_kokkos, build_polly)
        else:
            run_bench(kernel_dir, kernels, output_dir,
                      build_kokkos, build_polly, args.nb_iteration)


if __name__ == "__main__":
    main()
