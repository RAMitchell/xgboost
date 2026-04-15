from __future__ import annotations

import contextlib
import importlib.util
import io
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "run_clang_tidy", ROOT / "ops" / "script" / "run_clang_tidy.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Args:
    cpp = 1
    cuda = 1
    use_dmlc_gtest = True
    cuda_archs = ["70"]
    tidy_version = None
    build_dir = "build-clang-tidy-cuda"
    jobs = 2


class TestRunClangTidy(unittest.TestCase):
    def setUp(self) -> None:
        self.resolve_executable = mock.patch.object(
            MODULE, "resolve_executable", side_effect=lambda name: f"/mock/bin/{name}"
        )
        self.which = mock.patch.object(
            MODULE.shutil,
            "which",
            side_effect=lambda name: f"/mock/bin/{name}"
            if name in {"run-clang-tidy", "clang++"}
            else None,
        )
        self.resolve_executable.start()
        self.which.start()
        self.addCleanup(self.resolve_executable.stop)
        self.addCleanup(self.which.stop)
        with contextlib.redirect_stdout(io.StringIO()):
            self.linter = MODULE.ClangTidy(Args())

    def test_repo_warning_found_accepts_worktree_paths(self) -> None:
        output = (
            f"{ROOT}/src/common/timer.cc:10:3: warning: test\\n"
            "src/common/timer.cc:20:4: warning: another test\\n"
        )
        self.assertTrue(MODULE.repo_warning_found(output, ROOT))

    def test_normalize_cuda_arguments(self) -> None:
        args = [
            "/usr/local/cuda/bin/nvcc",
            "-forward-unknown-to-host-compiler",
            "-ccbin=/tmp/fake-g++",
            "-DDMLC_USE_CXX14=1",
            "-I/tmp/include",
            "-isystem",
            "/usr/local/cuda/targets/x86_64-linux/include",
            "-std=c++17",
            "--generate-code=arch=compute_70,code=[sm_70]",
            "--generate-code=arch=compute_70,code=[compute_70]",
            "-Xcompiler=-fopenmp",
            "--expt-extended-lambda",
            "--expt-relaxed-constexpr",
            "-Xfatbin=-compress-all",
            "--default-stream",
            "per-thread",
            "-x",
            "cu",
            "-c",
            "/tmp/test.cu",
            "-o",
            "test.o",
        ]

        normalized = self.linter._normalize_cuda_arguments(args)

        self.assertEqual(Path(normalized[0]).name, "clang++")
        self.assertIn("--cuda-path=/usr/local/cuda", normalized)
        self.assertIn("--cuda-gpu-arch=sm_70", normalized)
        self.assertIn("-x", normalized)
        self.assertIn("cuda", normalized)
        self.assertIn("-fopenmp", normalized)
        self.assertNotIn("--default-stream", normalized)
        self.assertNotIn("per-thread", normalized)
        self.assertNotIn("--expt-extended-lambda", normalized)
        self.assertNotIn("--expt-relaxed-constexpr", normalized)
        self.assertFalse(any(arg.startswith("--generate-code") for arg in normalized))


if __name__ == "__main__":
    unittest.main()
