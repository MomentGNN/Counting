# Counting Graph Substructures with Graph Neural Networks

This repository contains code for the paper *Counting Graph Substructures with Graph Neural Networks* submitted at Neurips 2023 

Abstract:

Graph Neural Networks (GNNs) are powerful representation learning tools that have achieved remarkable performance in various important tasks. However, their ability to count substructures, which play a crucial role in biological and social networks, remains uncertain. In this work, we fill this gap and characterize the representation power of GNNs in terms of their ability to produce powerful representations that count graph substructures. In particular, we study the message-passing operations of GNNs with random stationary input and show that they can produce permutation equivariant representations that are associated with high-order statistical moments. Using these representations, we prove that GNNs can learn how to count the number of 3- to 7-node cycles in a graph, and the number of 3- or 4-node cliques. We also prove that GNNs can count the number of connected components in a graph. To validate our theoretical findings, we conduct extensive experiments using synthetic and real-world molecular graphs. The results not only corroborate our theory but also reveal that GNNs are able to count cycles of up to 10 nodes.

## Code overview

This repository contains the source code that evaluates the performance of Moment-GNN in:

  - Cycle detection
  - Cycle counting

Source code for cycle detection is adapted from [https://github.com/cvignac/SMP](https://github.com/cvignac/SMP).
Source code for cycle counting is adapted from [https://github.com/gbouritsas/GSN](https://github.com/gbouritsas/GSN).


## Dependencies
[https://pytorch-geometric.readthedocs.io/en/latest/](Pytorch geometric) v1.6.1 was used. Please follow the instructions on the
website, as simple installations via pip do not work. In particular, the version of pytorch used must match the one of torch-geometric.
To run this code please install Python, PyTorch, and PyTorch-geometric (PyG).
We have used python 3.11, Pytorch 2.0, and PyG 2.3. 

Other required dependencies include pyyaml, and edict.


## Cycle detection

```
python3 .py
```

## Cycle counting
run
```
python3 main_penta.py
```
```
python3 main_hexa.py
```




## License
MIT
