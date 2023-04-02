import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = fh.readlines()
    for i, r in enumerate(requirements):
        requirements[i] = r.replace("\n", "")

#requirements.append("pylgr")

if __name__ == "__main__":
    setuptools.setup(
        name="optimalcontrol",
        version="0.6.0",
        description="Benchmark problems for optimal feedback control",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Tenavi Nakamura-Zimmerer",
        author_email="tenavi.nakamura-zimmerer@nasa.gov",
        packages=["optimalcontrol"],
        install_requires=requirements
    )
