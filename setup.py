from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension(
    "entropia",
    ["entropia\entropia.pyx"],
    extra_compile_args = ["-ffast-math"]
    )
]

setup(
    name = 'entropia',
    version = '0.2',
    author = "Barberia Juan Luis",
    author_email = "jbarberia@frba.utn.edu.ar",
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
