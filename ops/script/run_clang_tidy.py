#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from multiprocessing import Pool, cpu_count
from pathlib import Path
from time import time
from typing import Any


def resolve_executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(
            f"Required executable `{name}` is not on PATH. "
            "Use the xgboost conda env or install the missing LLVM tool."
        )
    return path


def load_command_arguments(entry: dict[str, Any]) -> list[str]:
    if "arguments" in entry:
        return list(entry["arguments"])
    return shlex.split(entry["command"])


def run_command(args: list[str]) -> tuple[int, str, list[str]]:
    completed = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return completed.returncode, completed.stdout, args


def repo_warning_found(output: str, root_path: Path) -> bool:
    pattern = re.compile(
        rf"^(?:{re.escape(str(root_path))}/)?(?:ops|src|tests|include)/.*warning:",
        re.MULTILINE,
    )
    return pattern.search(output) is not None


class ClangTidy:
    """clang-tidy wrapper.

    Args:
      args: Command line arguments.
          cpp_lint: Run linter on C++ source code.
          cuda_lint: Run linter on CUDA source code.
          use_dmlc_gtest: Whether to use gtest bundled in dmlc-core.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.cpp_lint = args.cpp
        self.cuda_lint = args.cuda
        self.use_dmlc_gtest: bool = args.use_dmlc_gtest
        self.cuda_archs = args.cuda_archs.copy() if args.cuda_archs else []
        self.jobs = args.jobs if args.jobs else min(cpu_count(), 35)

        if args.tidy_version:
            self.exe_name = "clang-tidy-" + str(args.tidy_version)
        else:
            self.exe_name = "clang-tidy"
        self.exe = resolve_executable(self.exe_name)
        self.run_clang_tidy = shutil.which("run-clang-tidy")
        self.clangxx = shutil.which("clang++")
        self.cmake = resolve_executable("cmake")

        print("Run linter on CUDA: ", self.cuda_lint)
        print("Run linter on C++:", self.cpp_lint)
        print("Use dmlc gtest:", self.use_dmlc_gtest)
        print("CUDA archs:", " ".join(self.cuda_archs))
        print("Parallel jobs:", self.jobs)

        if not self.cpp_lint and not self.cuda_lint:
            raise ValueError("Both --cpp and --cuda are set to 0.")

        self.root_path = Path(os.path.abspath(os.path.curdir))
        self.tidy_file = self.root_path / ".clang-tidy"
        self.generated_cdb = args.build_dir is None
        if args.build_dir is None:
            self.cdb_path = self.root_path / "cdb"
        else:
            self.cdb_path = Path(args.build_dir).resolve()
        self.normalized_cdb_dir: tempfile.TemporaryDirectory[str] | None = None
        self.normalized_cdb_path: Path | None = None
        self.files: list[str] = []

        print("Project root:", self.root_path)
        print("clang-tidy:", self.exe)
        if self.run_clang_tidy is not None:
            print("run-clang-tidy:", self.run_clang_tidy)
        else:
            print("run-clang-tidy: not found, will fall back to per-file mode.")
        print("Compilation database:", self.cdb_path)

    def __enter__(self) -> "ClangTidy":
        self.start = time()
        if self.generated_cdb:
            if self.cdb_path.exists():
                shutil.rmtree(self.cdb_path)
            self._generate_cdb()
        else:
            self._validate_cdb(self.cdb_path)
        self._normalize_cdb()
        return self

    def __exit__(self, *args: list[Any]) -> None:
        if self.generated_cdb and self.cdb_path.exists():
            shutil.rmtree(self.cdb_path)
        if self.normalized_cdb_dir is not None:
            self.normalized_cdb_dir.cleanup()
        self.end = time()
        print("Finish running clang-tidy:", self.end - self.start)

    def _validate_cdb(self, cdb_path: Path) -> None:
        cdb_file = cdb_path / "compile_commands.json"
        if not cdb_file.exists():
            raise FileNotFoundError(
                f"Could not find `{cdb_file}`. "
                "Pass a configured build directory with `--build-dir` or let the script "
                "generate one."
            )

    def _generate_cdb(self) -> None:
        """Run CMake to generate compilation database."""
        self.cdb_path.mkdir()
        cmake_args = [
            self.cmake,
            str(self.root_path),
            "-GNinja",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            "-DGOOGLE_TEST=ON",
            "-DCMAKE_CXX_FLAGS=-Wno-clang-diagnostic-deprecated-declarations",
        ]
        if self.use_dmlc_gtest:
            cmake_args.append("-DUSE_DMLC_GTEST=ON")
        else:
            cmake_args.append("-DUSE_DMLC_GTEST=OFF")

        if self.cuda_lint:
            cmake_args.extend(["-DUSE_CUDA=ON", "-DUSE_NCCL=ON"])
            if self.cuda_archs:
                arch_list = ";".join(self.cuda_archs)
                cmake_args.append(f"-DCMAKE_CUDA_ARCHITECTURES={arch_list}")

        subprocess.run(cmake_args, cwd=self.cdb_path, check=True)

    def _should_lint(self, path: Path) -> bool:
        if path.suffix == ".cc" and not self.cpp_lint:
            return False
        if path.suffix == ".cu" and not self.cuda_lint:
            return False
        if path.suffix not in {".cc", ".cu"}:
            return False
        return "dmlc-core" not in path.parts

    def _resolve_entry_path(self, entry: dict[str, Any]) -> Path:
        path = Path(entry["file"])
        if path.is_absolute():
            return path
        return (Path(entry["directory"]) / path).resolve()

    def _parse_cuda_arch(self, token: str) -> str | None:
        sm = re.search(r"sm_(\d+)", token)
        if sm is not None:
            return sm.group(1)
        compute = re.search(r"compute_(\d+)", token)
        if compute is not None:
            return compute.group(1)
        return None

    def _extend_host_flags(self, dest: list[str], value: str) -> None:
        for flag in value.split(","):
            flag = flag.strip()
            if flag:
                dest.append(flag)

    def _normalize_cuda_arguments(self, args: list[str]) -> list[str]:
        if "nvcc" not in Path(args[0]).name:
            return args
        if self.clangxx is None:
            raise RuntimeError(
                "CUDA clang-tidy normalization requires `clang++` on PATH. "
                "Install `clangxx` in the active environment."
            )

        nvcc_path = Path(args[0])
        cuda_path: Path | None = None
        if nvcc_path.parent.name == "bin":
            cuda_path = nvcc_path.parent.parent

        rest: list[str] = []
        archs: list[str] = []
        seen_archs: set[str] = set()

        i = 1
        while i < len(args):
            arg = args[i]

            if arg in {"-lineinfo", "-rdynamic", "-forward-unknown-to-host-compiler"}:
                i += 1
                continue

            if arg == "-ccbin":
                i += 2
                continue
            if arg.startswith("-ccbin="):
                i += 1
                continue

            if arg in {"-Xcompiler", "--compiler-options"}:
                if i + 1 < len(args):
                    self._extend_host_flags(rest, args[i + 1])
                i += 2
                continue
            if arg.startswith("-Xcompiler=") or arg.startswith("--compiler-options="):
                self._extend_host_flags(rest, arg.split("=", 1)[1])
                i += 1
                continue

            if arg in {"--default-stream", "-default-stream"}:
                i += 2
                continue
            if arg.startswith("--default-stream="):
                i += 1
                continue

            if arg.startswith("--expt-") or arg == "-Xfatbin=-compress-all":
                i += 1
                continue

            if arg in {"--generate-code", "-gencode"}:
                if i + 1 < len(args):
                    arch = self._parse_cuda_arch(args[i + 1])
                    if arch is not None and arch not in seen_archs:
                        seen_archs.add(arch)
                        archs.append(arch)
                i += 2
                continue
            if arg.startswith("--generate-code=") or arg.startswith("-gencode="):
                arch = self._parse_cuda_arch(arg)
                if arch is not None and arch not in seen_archs:
                    seen_archs.add(arch)
                    archs.append(arch)
                i += 1
                continue

            if arg == "-x" and i + 1 < len(args) and args[i + 1] == "cu":
                rest.extend(["-x", "cuda"])
                i += 2
                continue

            if arg.startswith("-isystem="):
                rest.extend(["-isystem", arg.split("=", 1)[1]])
                i += 1
                continue

            rest.append(arg)
            i += 1

        if not archs:
            archs = self.cuda_archs.copy()

        normalized = [self.clangxx]
        if cuda_path is not None:
            normalized.append(f"--cuda-path={cuda_path}")
        normalized.extend([f"--cuda-gpu-arch=sm_{arch}" for arch in archs])
        normalized.extend(rest)
        return normalized

    def _normalize_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve_entry_path(entry)
        arguments = load_command_arguments(entry)
        if path.suffix == ".cu":
            arguments = self._normalize_cuda_arguments(arguments)

        return {
            "directory": entry["directory"],
            "file": str(path),
            "arguments": arguments,
        }

    def _normalize_cdb(self) -> None:
        cdb_file = self.cdb_path / "compile_commands.json"
        with open(cdb_file, "r", encoding="utf-8") as fd:
            compile_commands = json.load(fd)

        self.files = []
        normalized_commands = []
        for entry in compile_commands:
            path = self._resolve_entry_path(entry)
            if not self._should_lint(path):
                continue
            self.files.append(str(path))
            normalized_commands.append(self._normalize_entry(entry))

        if not normalized_commands:
            raise RuntimeError(
                "No source files matched the requested clang-tidy filters."
            )

        self.normalized_cdb_dir = tempfile.TemporaryDirectory(
            prefix="xgboost-clang-tidy-"
        )
        self.normalized_cdb_path = Path(self.normalized_cdb_dir.name)
        with open(
            self.normalized_cdb_path / "compile_commands.json", "w", encoding="utf-8"
        ) as fd:
            json.dump(normalized_commands, fd, indent=2)

        print("Normalized compile commands:", self.normalized_cdb_path)
        print("Selected translation units:", len(self.files))

    def _header_filter(self) -> str:
        src = str(self.root_path / "src").replace("/", "\\/")
        include = str(self.root_path / "include").replace("/", "\\/")
        return f"({src}|{include})"

    def _run_with_run_clang_tidy(self) -> tuple[int, str]:
        assert self.normalized_cdb_path is not None
        assert self.run_clang_tidy is not None
        args = [
            self.run_clang_tidy,
            "-clang-tidy-binary",
            self.exe,
            "-p",
            str(self.normalized_cdb_path),
            "-j",
            str(self.jobs),
            "-config-file",
            str(self.tidy_file),
            "-header-filter",
            self._header_filter(),
        ]
        completed = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        print(completed.stdout, end="")
        return completed.returncode, completed.stdout

    def _run_with_pool(self) -> tuple[int, str]:
        assert self.normalized_cdb_path is not None
        common = [
            f"-p={self.normalized_cdb_path}",
            f"--config-file={self.tidy_file}",
            f"--header-filter={self._header_filter()}",
        ]
        commands = [[self.exe, *common, path] for path in self.files]
        output = []
        proc_code = 0
        with Pool(self.jobs) as pool:
            for returncode, msg, args in pool.map(run_command, commands):
                proc_code = max(proc_code, returncode)
                if msg:
                    print(msg, end="")
                    output.append(msg)
        return proc_code, "".join(output)

    def smoke_test(self) -> None:
        test_file_path = self.root_path / "ops" / "script" / "test_tidy.cc"
        cmd = [self.exe, f"--config-file={self.tidy_file}", str(test_file_path)]
        proc_code, output, _ = run_command(cmd)
        if proc_code != 0 or "warning:" not in output:
            raise RuntimeError(output)
        print("clang-tidy is working.")

    def run(self) -> bool:
        """Run clang-tidy."""
        if self.run_clang_tidy is not None:
            process_status, output = self._run_with_run_clang_tidy()
        else:
            process_status, output = self._run_with_pool()

        passed = process_status == 0 and not repo_warning_found(output, self.root_path)
        if not passed:
            print(
                "Errors in `thrust` namespace can be safely ignored.",
                "Please address rest of the clang-tidy warnings.",
            )
        return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run clang-tidy.")
    parser.add_argument("--cpp", type=int, default=1)
    parser.add_argument(
        "--tidy-version",
        type=int,
        default=None,
        help="Specify the version of preferred clang-tidy.",
    )
    parser.add_argument("--cuda", type=int, default=1)
    parser.add_argument(
        "--use-dmlc-gtest",
        action="store_true",
        help="Whether to use gtest bundled in dmlc-core.",
    )
    parser.add_argument(
        "--cuda-archs", action="append", help="List of CUDA archs to build"
    )
    parser.add_argument(
        "--build-dir",
        type=str,
        default=None,
        help="Reuse an existing build directory with `compile_commands.json` instead "
        "of generating a temporary one.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        help="Number of parallel clang-tidy jobs. Defaults to `min(cpu_count(), 35)`.",
    )
    args = parser.parse_args()

    linter = ClangTidy(args)
    linter.smoke_test()
    with linter:
        passed = linter.run()
    if not passed:
        sys.exit(1)
