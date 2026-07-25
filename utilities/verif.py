import os
import re
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
                        choices=["MINI", "SMALL", "MEDIUM",
                                 "LARGE", "EXTRALARGE"],
                        help="Dataset size")
    parser.add_argument("--cxx_compiler", type=str, required=True,
                        help="C++ compiler")
    parser.add_argument("--cxx_compiler_polly_vanilla", type=str, default="",
                        required=False,
                        help="C++ compiler for polly vanilla")
    parser.add_argument("--kokkos_install_dir", type=str, required=True,
                        help="Install directory for Kokkos")
    parser.add_argument("--polybench_dir", type=str, required=True,
                        help="Polybench directory")
    parser.add_argument("--process_dir", type=str, required=True,
                        help="Polybench execution directory")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="isl",
        choices=["none", "isl", "pluto", "ppcg"],
        help="Choose the scheduler to use for Polly"
    )
    args = parser.parse_args()
    return args


def display_row_line(verif, polly_vanilla):
    if verif:
        if (polly_vanilla):
            print("\r\033[K+" + "-" * 27 + "+" + "-" * 27 + "+" + "-" * 27 +
                  "+" + "-" * 27 + "+" + "-" * 27 + "-" * 17 + "+")
        else:
            print("\r\033[K+" + "-" * 27 + "+" + "-" * 27 +
                  "+" + "-" * 27 + "+" + "-" * 27 + "-" * 17 + "+")
    else:
        if polly_vanilla:
            print("\r\033[K+" + "-" * 27 + "+" + "-" * 27 + "+" + "-" * 27 +
                  "+" + "-" * 27 + "+" + "-" * 27 + "+")
        else:
            print("\r\033[K+" + "-" * 27 + "+" + "-" * 27 +
                  "+" + "-" * 27 + "+")


def display_row_title(verif, polly_vanilla, kernel):
    display_row_line(verif, polly_vanilla)
    if verif:
        if polly_vanilla:
            print((f"| {kernel.center(25)} | {'Standard'.center(25)} | "
                   f"{'Vanilla'.center(25)} | "
                   f"{'Kokkos'.center(25)} | {'Polly'.center(25)} | "
                   f"{'Verif'.center(14)} |"))
        else:
            print((f"| {kernel.center(25)} | {'Standard'.center(25)} | "
                   f"{'Kokkos'.center(25)} | {'Polly'.center(25)} | "
                   f"{'Verif'.center(14)} |"))
    else:
        if polly_vanilla:
            print((f"| {kernel.center(25)} | {'Vanilla'.center(25)} |"
                   f"{'Kokkos'.center(26)} | {'V/Polly'.center(25)} | "
                   f"{'K/Polly'.center(25)} |"))
        else:
            print((f"| {kernel.center(25)} | "
                   f"{'Kokkos'.center(25)} | {'Polly'.center(25)} |"))
    display_row_line(verif, polly_vanilla)


def display_row_data(verif, polly_vanilla,
                     kernel,
                     time_std,
                     time_vanilla,
                     time_kokkos,
                     time_polly,
                     check_str):
    if verif:
        if polly_vanilla:
            print((f"| {kernel.center(25)} | "
                   f"{str(time_std).center(25)} | "
                   f"{str(time_vanilla).center(25)} | "
                   f"{str(time_kokkos).center(25)} | "
                   f"{str(time_polly).center(25)} | "
                   f"{check_str.center(47)} |"))
        else:
            print((f"| {kernel.center(25)} | "
                   f"{str(time_std).center(25)} | "
                   f"{str(time_kokkos).center(25)} | "
                   f"{str(time_polly).center(25)} | "
                   f"{check_str.center(36)} |"))
        display_row_line(verif, polly_vanilla)
    else:
        if polly_vanilla:
            print((f"| {str(kernel).center(25)} | "
                   f"{str(time_vanilla).center(25)} | "
                   f"{str(time_kokkos).center(25)} | "
                   f"{str(time_polly).center(25)} |"))
        else:
            print((f"| {str(kernel).center(25)} | "
                   f"{str(time_kokkos).center(25)} | "
                   f"{str(time_polly).center(25)} |"))
        display_row_line(verif, polly_vanilla)


def display_row_data_bench(kernel,
                           statistics_polly_vanilla,
                           statistics_kokkos,
                           statistics_polly,
                           polly_vanilla):
    avg_v = med_v = std_dev_v = min_v = max_v = 0
    if polly_vanilla:
        avg_v, med_v, std_dev_v, min_v, max_v = statistics_polly_vanilla
    avg_k, med_k, std_dev_k, min_k, max_k = statistics_kokkos
    avg_p, med_p, std_dev_p, min_p, max_p = statistics_polly
    speedup_kp = avg_k / avg_p if avg_p != 0 else float('inf')
    speedup_vp = 1
    if polly_vanilla:
        speedup_vp = avg_v / avg_p if avg_p != 0 else float('inf')

    if polly_vanilla:
        print((f"| {COLOR[GREEN]}{str(kernel).center(25)}{COLOR[NO_COLOR]} | "
               f"{''.center(25)} | {''.center(25)} | {''.center(25)} | "
               f"{''.center(25)} |"))
        print((f"| {'speedup'.center(25)} | {str(f'{1}').center(25)} | "
               f"{str(f'{1}').center(25)} | "
               f"{str(f'{speedup_vp:,.2f}').center(25)} | "
               f"{str(f'{speedup_kp:,.2f}').center(25)} |"))
        print((f"| {'average'.center(25)} | "
               f"{str(f'{avg_v:,.1f}').center(25)} | "
               f"{str(f'{avg_k:,.1f}').center(25)} | "
               f"{str(f'{avg_p:,.1f}').center(25)} | "
               f"{str(f'{avg_p:,.1f}').center(25)} |"))
        print((f"| {'median'.center(25)} | "
               f"{str(f'{med_v:,.1f}').center(25)} | "
               f"{str(f'{med_k:,.1f}').center(25)} | "
               f"{str(f'{med_p:,.1f}').center(25)} | "
               f"{str(f'{med_p:,.1f}').center(25)} |"))
        print((f"| {'standard deviation'.center(25)} | "
               f"{str(f'{std_dev_v:,.1f}').center(25)} | "
               f"{str(f'{std_dev_k:,.1f}').center(25)} | "
               f"{str(f'{std_dev_p:,.1f}').center(25)} | "
               f"{str(f'{std_dev_p:,.1f}').center(25)} |"))
        print((f"| {'minimum'.center(25)} | "
               f"{str(f'{min_v:,.1f}').center(25)} | "
               f"{str(f'{min_k:,.1f}').center(25)} | "
               f"{str(f'{min_p:,.1f}').center(25)} | "
               f"{str(f'{min_p:,.1f}').center(25)} |"))
        print((f"| {'max'.center(25)} | {str(f'{max_v:,.1f}').center(25)} | "
               f"{str(f'{max_k:,.1f}').center(25)} | "
               f"{str(f'{max_p:,.1f}').center(25)} | "
               f"{str(f'{max_p:,.1f}').center(25)} |"))
    else:
        print((f"| {COLOR[GREEN]}{str(kernel).center(25)}"
               f"{COLOR[NO_COLOR]} | {''.center(25)} | {''.center(25)} |"))
        print((f"| {'speedup'.center(25)} | {str(f'{1}').center(25)} | "
               f"{str(f'{speedup_kp:,.2f}').center(25)} |"))
        print((f"| {'average'.center(25)} | {str(f'{avg_k:,.1f}').center(25)} | "
               f"{str(f'{avg_p:,.1f}').center(25)} |"))
        print((f"| {'median'.center(25)} | {str(f'{med_k:,.1f}').center(25)} | "
               f"{str(f'{med_p:,.1f}').center(25)} |"))
        print((f"| {'standard deviation'.center(25)} | "
               f"{str(f'{std_dev_k:,.1f}').center(25)} | "
               f"{str(f'{std_dev_p:,.1f}').center(25)} |"))
        print((f"| {'minimum'.center(25)} | {str(f'{min_k:,.1f}').center(25)} | "
               f"{str(f'{min_p:,.1f}').center(25)} |"))
        print((f"| {'max'.center(25)} | {str(f'{max_k:,.1f}').center(25)} | "
               f"{str(f'{max_p:,.1f}').center(25)} |"))
    display_row_line(False, polly_vanilla)


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
    standard_deviation = statistics.stdev(values) if len(values) != 1 else 0
    minimum = min(values)
    maximum = max(values)
    return (average, median, standard_deviation, minimum, maximum)


def sanitize_zeros(filepath):
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        clean_content = re.sub(r'-(0\.0+(?!\d))', r'\1', content)

        if content != clean_content:
            with open(filepath, 'w') as f:
                f.write(clean_content)

        return clean_content

    except IOError as e:
        print(f"Erreur lors du traitement de {filepath}: {e}")
        return None


def compute_hash(fichier):
    hasher = hashlib.sha256()
    with open(fichier, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def check_output(file_std, file_polly_vanilla, file_kokkos, file_polly,
                 polly_vanilla):
    sanitize_zeros(file_std)
    if polly_vanilla:
        sanitize_zeros(file_polly_vanilla)
    sanitize_zeros(file_kokkos)
    sanitize_zeros(file_polly)

    hash_std = compute_hash(file_std)
    hash_vanilla = compute_hash(file_polly_vanilla) if polly_vanilla else None
    hash_kokkos = compute_hash(file_kokkos)
    hash_polly = compute_hash(file_polly)

    res = ""
    if polly_vanilla:
        if hash_std != hash_vanilla:
            res += f"{COLOR[RED]}V{COLOR[NO_COLOR]}"
        else:
            res += f"{COLOR[GREEN]}V{COLOR[NO_COLOR]}"
    if hash_std != hash_kokkos:
        res += f"{COLOR[RED]}K{COLOR[NO_COLOR]}"
    else:
        res += f"{COLOR[GREEN]}K{COLOR[NO_COLOR]}"
    if hash_std != hash_polly:
        res += f"{COLOR[RED]}P{COLOR[NO_COLOR]}"
    else:
        res += f"{COLOR[GREEN]}P{COLOR[NO_COLOR]}"
    return res


def generate_build_file(polybench_dir,
                        output_dir,
                        build_std,
                        build_polly_vanilla,
                        build_kokkos,
                        build_polly,
                        cxx_compiler,
                        cxx_compiler_polly_vanilla,
                        kokkos_install_dir,
                        dataset,
                        verif,
                        scheduler):
    print_output = "ON" if verif else "OFF"

    cmake_command_base = (
        f"cmake -S {polybench_dir} "
        f"-DCMAKE_CXX_COMPILER={cxx_compiler} "
        f"-DCMAKE_BUILD_TYPE=Release "
        f"-DPB_CYCLE_MONITORING=ON "
        f"-DPB_DUMP_ARRAYS={print_output} "
        f"-DPB_DATASET_SIZE={dataset} ")

    # build standard version
    if verif:
        print(f"{COLOR[GREEN]}Building standard version"
              f"{COLOR[NO_COLOR]}\r", end="")
        cmake_command_standard = cmake_command_base + f"-B {build_std}"
        run_command(cmake_command_standard, os.path.join(
            output_dir, "cmake_standard.log"))

    # build standard version with polly vanilla
    if (cxx_compiler_polly_vanilla != ""):
        print(f"{COLOR[GREEN]}Building standard version with polly vanilla"
              f"{COLOR[NO_COLOR]}\r", end="")
        cmake_command_polly_vanilla = (
            f"cmake -S {polybench_dir} "
            f"-B {build_polly_vanilla} "
            f"-DCMAKE_CXX_COMPILER={cxx_compiler_polly_vanilla} "
            f"-DCMAKE_BUILD_TYPE=Release "
            f"-DPB_CYCLE_MONITORING=ON "
            f"-DPB_DUMP_ARRAYS={print_output} "
            f"-DPB_DATASET_SIZE={dataset} "
            f"-DPB_USE_POLLY=ON "
            f"-DPB_USE_VANILLA_POLLY=ON ")
        run_command(cmake_command_polly_vanilla, os.path.join(
            output_dir, "cmake_polly_vanilla.log"))

    # build kokkos version
    print(f"\r\033[K\r{COLOR[GREEN]}Building Kokkos version"
          f"{COLOR[NO_COLOR]}", end="")
    cmake_command_kokkos = cmake_command_base + (f"-B "
                                                 f"{build_kokkos} "
                                                 f"-DPB_KOKKOS=ON "
                                                 f"-DPB_KOKKOS_DIR="
                                                 f"{kokkos_install_dir} "
                                                 f"-DPB_POLLY_SCHEDULER={
                                                     scheduler} "
                                                 f"-DKokkos_ENABLE_SERIAL=ON "
                                                 f"-DKokkos_ENABLE_OPENMP=ON ")
    run_command(cmake_command_kokkos, os.path.join(
        output_dir, "cmake_kokkos.log"))

    # build polly version
    print(f"\r\033[K\r{COLOR[GREEN]}Building Polly version"
          f"{COLOR[NO_COLOR]}\r", end="")
    cmake_command_polly = cmake_command_base + (f"-B {build_polly} "
                                                f"-DPB_KOKKOS=ON "
                                                f"-DPB_KOKKOS_DIR="
                                                f"{kokkos_install_dir} "
                                                f"-DPB_USE_POLLY=ON "
                                                f"-DPB_POLLY_SCHEDULER={
                                                    scheduler} "
                                                f"-DKokkos_ENABLE_SERIAL=ON ")
    run_command(cmake_command_polly, os.path.join(
        output_dir, "cmake_polly.log"))


def run_verif(kernel_dir,
              kernels,
              output_dir,
              build_std,
              build_polly_vanilla,
              build_kokkos,
              build_polly,
              polly_vanilla):
    for kernel in kernels:
        kernel_output_path = f"{output_dir}/{kernel_dir}/{kernel}"
        os.makedirs(kernel_output_path, exist_ok=True)

        versions = []
        if polly_vanilla:
            versions = [(build_std, "std"),
                        (build_polly_vanilla, "vanilla"),
                        (build_kokkos, "kokkos"),
                        (build_polly, "polly")]
        else:
            versions = [(build_std, "std"),
                        (build_kokkos, "kokkos"),
                        (build_polly, "polly")]

        for build, version in versions:
            os.chdir(build)
            print(f"{COLOR[YELLOW]}Building {kernel} "
                  f"{version} version{COLOR[NO_COLOR]}\r", end="")
            make_command = f"make -j {kernel}"
            run_command(make_command, os.path.join(
                kernel_output_path, f"{kernel}_{version}.compile"))
            print(f"{COLOR[YELLOW]}\rRunning {kernel} "
                  f"{version} version{COLOR[NO_COLOR]}\r", end="")
            exec_command = (
                f"{ARGS_ENV} {build}/{kernel_dir}/"
                f"{kernel}/{kernel}"
            )

            run_command(exec_command,
                        os.path.join(kernel_output_path,
                                     f"{kernel}_{version}.time"),
                        os.path.join(kernel_output_path,
                                     f"{kernel}_{version}.out"))

        check_str = check_output(os.path.join(kernel_output_path,
                                              f"{kernel}_std.out"),
                                 os.path.join(kernel_output_path,
                                              f"{kernel}_vanilla.out"),
                                 os.path.join(kernel_output_path,
                                              f"{kernel}_kokkos.out"),
                                 os.path.join(kernel_output_path,
                                              f"{kernel}_polly.out"),
                                 polly_vanilla)
        display_row_data(True, polly_vanilla, kernel, 1, 1, 1, 1, check_str)


def run_bench(kernel_dir,
              kernels,
              output_dir,
              build_polly_vanilla,
              build_kokkos,
              build_polly,
              nb_iteration,
              polly_vanilla):
    for kernel in kernels:
        kernel_output_path = f"{output_dir}/{kernel_dir}/{kernel}"
        os.makedirs(kernel_output_path, exist_ok=True)
        statistics = []
        versions = []
        if polly_vanilla:
            versions = [(build_polly_vanilla, "vanilla"),
                        (build_kokkos, "kokkos"),
                        (build_polly, "polly")]
        else:
            versions = [(build_kokkos, "kokkos"),
                        (build_polly, "polly")]

        for build, version in versions:
            os.chdir(build)
            print(f"\r\033[K{COLOR[YELLOW]}Building {kernel} "
                  f"{version} version{COLOR[NO_COLOR]}", end="")
            make_command = f"make -j {kernel}"
            run_command(make_command, os.path.join(
                kernel_output_path, f"{kernel}_{version}.compile"))
            for i in range(nb_iteration):
                print(f"\r\033[K{COLOR[YELLOW]}Running {kernel} "
                      f"{version} version (iteration {i+1}/{nb_iteration})"
                      f"{COLOR[NO_COLOR]}", end="")
                exec_command = (
                    f"{ARGS_ENV} {build}/{kernel_dir}/"
                    f"{kernel}/{kernel}"
                )

                time_file = os.path.join(kernel_output_path,
                                         f"{kernel}_{version}.time")
                run_command(exec_command, time_file)
            statistics.append(do_statistics(time_file))
        print("\r\033[K", end="")

        stats_polly_vanilla = statistics[0] if polly_vanilla else None
        stats_kokkos = statistics[1] if polly_vanilla else statistics[0]
        stats_polly = statistics[2] if polly_vanilla else statistics[1]

        display_row_data_bench(
            kernel, stats_polly_vanilla, stats_kokkos, stats_polly,
            polly_vanilla)


def main():
    args = parse_args()

    polybench_dir = args.polybench_dir
    process_dir = args.process_dir
    build_std = os.path.join(process_dir, "build_std")
    build_polly_vanilla = os.path.join(process_dir, "build_polly_vanilla")
    build_kokkos = os.path.join(process_dir, "build_kokkos")
    build_polly = os.path.join(process_dir, "build_polly")
    output_dir = os.path.join(process_dir, "output")
    scheduler = args.scheduler

    mode = f"{args.nb_iteration} iterations" if not args.verif else "verif"

    print(
        f"Run {mode}\n"
        f"Compiler : {args.cxx_compiler}\n"
        f"Kokkos : {args.kokkos_install_dir}\n"
        f"Dataset : {args.dataset}\n"
        f"scheduler : {scheduler}\n"
        f"Output directory : {process_dir}"
    )

    if os.path.exists(process_dir):
        shutil.rmtree(process_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(build_std, exist_ok=True)
    os.makedirs(build_polly_vanilla, exist_ok=True)
    os.makedirs(build_kokkos, exist_ok=True)
    os.makedirs(build_polly, exist_ok=True)

    datasets = {
        "linear-algebra/blas": ["gemm"],
    }

    generate_build_file(polybench_dir, output_dir,
                        build_std, build_polly_vanilla, build_kokkos,
                        build_polly, args.cxx_compiler,
                        args.cxx_compiler_polly_vanilla,
                        args.kokkos_install_dir, args.dataset, args.verif,
                        scheduler)

    for kernel_dir, kernels in datasets.items():
        polly_vanilla = args.cxx_compiler_polly_vanilla != ""
        display_row_title(args.verif, polly_vanilla, kernel_dir)
        if args.verif:
            run_verif(kernel_dir, kernels, output_dir,
                      build_std, build_polly_vanilla, build_kokkos,
                      build_polly, polly_vanilla)
        else:
            run_bench(kernel_dir, kernels, output_dir, build_polly_vanilla,
                      build_kokkos, build_polly, args.nb_iteration,
                      polly_vanilla)


if __name__ == "__main__":
    main()
