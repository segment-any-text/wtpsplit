from setuptools import setup

setup(
    name="wtpsplit",
    version="2.0.0",
    packages=["wtpsplit"],
    description="Universal Robust, Efficient and Adaptable Sentence Segmentation",
    author="Markus Frohmann, Igor Sterner, Benjamin Minixhofer",
    author_email="markus.frohmann@gmail.com",
    install_requires=[
        "onnxruntime>=1.13.1",
        "transformers>=4.22.2,<4.40",
        "numpy>=1.0,<2.0",
        "scikit-learn>=1",
        "tqdm",
        "skops",
        "pandas>=1",
        "cached_property",  # for Py37
        "torchinfo",
        "mosestokenizer",
        "adapters==0.2.1"
    ],
    url="https://github.com/segment-any-text/wtpsplit",
    package_data={"wtpsplit": ["data/*"]},
    include_package_data=True,
    license="MIT",
)
