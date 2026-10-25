from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension("models.other_models", ["models/other_models.py"]),
]

setup(
    name="model",
    version="0.1",
    description="models",
    packages=["models"],
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': "3"}
    ),
    zip_safe=False,
)
