import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="princoml",
    version="0.1.0",
    author="Jason M. Cherry",
    author_email="jcherry@gmail.com",
    description="Custom neural network code.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Calvinxc1/PrincoML",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy >= 1.18.0, < 2.0.0',
        'pandas >= 1.0.0, < 2.0.0',
        'torch >= 1.4.0, < 2.0.0',
        'tqdm >= 4.45.0, < 5.0.0',
        'matplotlib >= 3.2.0, < 4.0.0',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)