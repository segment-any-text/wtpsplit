from setuptools import setup

setup(
    name="wtpsplit",
    version="1.1.0",
    packages=["wtpsplit"],
    description="Robust, adaptible sentence segmentation for 85 languages",
    author="Benjamin Minixhofer",
    author_email="bminixhofer@gmail.com",
    install_requires=["torch", "transformers", "numpy", "scikit-learn", "tqdm", "skops"],
    url="https://github.com/bminixhofer/wtpsplit",
    package_data={"wtpsplit": ["data/*"]},
    include_package_data=True,
    license="MIT",
)
