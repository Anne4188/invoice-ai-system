"""
Docstring for training

为什么一定要有 __init__.py？

这是非常关键但容易被忽略的一步。

没有 __init__.py 会发生什么？

如果你之后跑：

python -m src.training.train_lora


Python 会报类似：

ModuleNotFoundError: No module named 'src.training'


因为 Python 只把“带 __init__.py 的目录”当成包（package）。
"""
