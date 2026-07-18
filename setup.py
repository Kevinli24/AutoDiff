import os
import sys
import warnings

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


compile_args = ["/std:c++17", "/O2"] if sys.platform == "win32" else ["-std=c++17", "-O3"]


class OptionalBuildExt(build_ext):
    """Allow installation to use the NumPy backend when no compiler exists."""

    def _handle_failure(self, error):
        if os.environ.get("AUTODIFF_REQUIRE_CPP") == "1":
            raise error
        warnings.warn(f"C++ extension was not built; using the Python backend: {error}")

    def run(self):
        try:
            super().run()
        except Exception as error:
            self._handle_failure(error)

setup(
    cmdclass={"build_ext": OptionalBuildExt},
    ext_modules=[
        Extension(
            "autodiff._tensor",
            sources=["src/python_bindings.cpp", "src/cpp/tensor.cpp"],
            include_dirs=["include"],
            language="c++",
            extra_compile_args=compile_args,
        )
    ]
)
