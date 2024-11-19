from setuptools import setup

setup(
    name="signDNE",
    version="0.1.0",
    package_dir={"": "src"},
    install_requires=[
        "scipy",
        "trimesh",
        "numpy",
        "pyvista",
        "pandas",
        "networkx",
        "rtree",
        "pyglet<2"
    ],
    entry_points={
        "console_scripts": [
            "signDNE=_signDNE_cli:main",
        ],
    },
    author="Felix Risbro Hjerrild, Shan Shan",
    author_email="fehje22@student.sdu.dk",
    description="A Python package for ariaDNE and its sign-oriented extension",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ariaDNE",  # Your project's URL
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
