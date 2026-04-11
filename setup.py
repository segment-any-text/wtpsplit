from setuptools import setup, find_packages

setup(
    name="wtpsplit",
    version="2.2.1",
    packages=find_packages(),
    description="Universal Robust, Efficient and Adaptable Sentence Segmentation",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Markus Frohmann, Igor Sterner, Benjamin Minixhofer",
    author_email="markus.frohmann@gmail.com",
    python_requires=">=3.9",
    install_requires=[
        # "onnxruntime>=1.13.1", # can make conflicts between onnxruntime and onnxruntime-gpu
        "transformers>=4.22.2",
        "huggingface-hub",
        "numpy>=1.0",
        "scikit-learn>=1",
        "tqdm",
        "skops",
        "pandas>=1",
        "mosestokenizer",
    ],
    extras_require={
        "adapters": ["adapters>=1.0.1"],
        "onnx-gpu": ["onnxruntime-gpu>=1.13.1"],
        "onnx-cpu": ["onnxruntime>=1.13.1"],
    },
    url="https://github.com/segment-any-text/wtpsplit",
    package_data={"wtpsplit": ["data/*"]},
    include_package_data=True,
    license="MIT",
)
