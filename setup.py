import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

if __name__ == "__main__":
    setuptools.setup(
        name="ocp",
        version="0.1.0",
        description="Benchmark problems for optimal feedback control",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Tenavi Nakamura-Zimmerer",
        author_email="tenavi.nakamura-zimmerer@nasa.gov",
        packages=["ocp"],
        install_requires=[
            "scipy>=1.5.2",
            "numpy>=1.19.1",
            "pandas>=1.4.4",
            "tqdm>=4.64.1",
            "pytest>=6.1.1",
            "matplotlib>=3.1.2",
            "pylgr"
        ]
    )
