import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fluidlearn", # Replace with your own username
    version="0.1.0",
    author="Manu Jayadharan",
    author_email="manu.jayadharan@pitt.edu",
    description="Package to solve fluid flow PDEs using deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mjayadharan/FluidLearn/archive/v0.1.0.tar.gz",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
