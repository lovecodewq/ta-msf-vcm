from setuptools import setup, find_packages

setup(
    name="image_compress",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.6",
    install_requires=[
        "torch",
        "torchvision",
        "pillow",
        "matplotlib",
        "pyyaml",
        "numpy",
        "compressai",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ],
    },
) 