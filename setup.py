from setuptools import setup

setup(
    name="wtpsplit",
    version="1.2.0",
    packages=["wtpsplit"],
    description="Robust, adaptible sentence segmentation for 85 languages",
    author="Benjamin Minixhofer",
    author_email="bminixhofer@gmail.com",
    install_requires=[
        "onnxruntime>=1.13.1",
        "transformers>=4.22.2",
        "numpy>=1",
        "scikit-learn>=1",
        "tqdm",
        "skops",
        "pandas>=1",
        "cached_property",  # for Py37
    ],
    url="https://github.com/bminixhofer/wtpsplit",
    package_data={"wtpsplit": ["data/*"]},
    include_package_data=True,
    license="MIT",
)
