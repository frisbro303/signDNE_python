from setuptools import setup

setup(
    name="sign_ariaDNE",
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
            "sign_ariaDNE=_sign_ariaDNE_cli:main",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python package for ariaDNE and its sign-oriented extension",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ariaDNE",  # Your project's URL
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
