## Autograd Engine
This project is a small custom Autograd engine built to further my knowledge of neural networks and machine learning.
This engine implements backpropagation over a dynamically built Directed Acyclic Graph (DAG). This project involved developing 
a small neural network’s library with a PyTorch-like API and included the capability for tracing and visualization using Graphviz visualizations.

### Training
The notebook `demo.ipynb` includes a full demo on training a multilayer neural network
(MLP) binary classifier. This is achieved by initializing a neural net from the `src.nn` module,
implementing a simple svm "max-margin" binary classification loss and using SGD for optimization.

### Tracing/Visualization
The notebook `trace_graph.ipynb` produces graphvis visualizations.

### Tests
The unit tests require the `PyTorch` dependency.
```
python -m pytest
```

