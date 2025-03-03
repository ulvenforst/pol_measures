from setuptools import setup, find_packages

setup(
    name="measures",
    version="0.1.0a2",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "mypy>=0.900",
        ],
    },
    python_requires=">=3.8",
    author="Juan Camilo Narváez Tascón",
    author_email="juan.narvaez.tascon@correounivalle.edu.co",
    description="A package for computing various polarization measures in PROMUEVA",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Ulvenforst/pol_measures",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
