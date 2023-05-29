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

To run this code please install Python, PyTorch, and PyTorch-geometric (PyG).

We have used python 3.11, Pytorch 2.0, and PyG 2.3. 

Other required dependencies include pyyaml, and edict.


## Cycle detection
To run the experiments for cycle detection in Section 6.2 please run:
```
python cycles_main_synth.py
```
Play with the parameters k, n, and generalization to reproduce all the experiments. In the config_cycles.yaml set num_layers : -1.

To run the experiments for cycle detection in the ZINC dataset, as in Section 6.3, please run:
```
python cycles_ZINC.py
```
for k=9 and k=10. To reproduce the reported results set num_layers : 2 for k=9 and num_layers : 3 for k=10 in the config_cycles.yaml file.

## Cycle counting
To count the 5- and 6-node cycles in the ZINC dataset run the following programms.
```
python main_penta.py
```
```
python main_hexa.py
```
## License
MIT
