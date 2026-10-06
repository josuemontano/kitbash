"""Build the constrained QEM extension for Poetry wheels and editable installs."""

import os
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Distribution, Extension
from setuptools.command.build_ext import build_ext


def build() -> None:
    source = Path("src/kitbash/retopology/triflow/geometry/_qem.pyx")
    extension = Extension(
        "kitbash.retopology.triflow.geometry._qem",
        [str(source)],
        language="c++",
        include_dirs=[str(source.parent)],
        depends=[str(source.with_name("Simplify.h"))],
        extra_compile_args=["/O2", "/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17"],
    )
    distribution = Distribution({
        "ext_modules": cythonize([extension], compiler_directives={"language_level": 3}),
        "package_dir": {"": "src"},
    })
    command = build_ext(distribution)
    command.ensure_finalized()
    command.inplace = True
    command.run()


if __name__ == "__main__":
    build()
