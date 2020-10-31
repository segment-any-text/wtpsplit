## Installation

NNSplit for Python has onnxruntime as the only dependency.

Install NNSplit with pip: 

```bash
pip install nnsplit
```
&nbsp;

To enable GPU support, install onnxruntime-gpu:

```bash
pip install onnxruntime-gpu
```
&nbsp;

## Use

```python
from nnsplit import NNSplit
splitter = NNSplit.load("en")

# returns `Split` objects
splits = splitter.split(["This is a test This is another test."])[0]

# a `Split` can be iterated over to yield smaller splits or stringified with `str(...)`.
for sentence in splits:
   print(sentence)
```