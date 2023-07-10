# ES-GNN
Source codes for "ES-GNN: Generalizing Graph Neural Networks Beyond Homophily with Edge Splitting".

## Abstract
While Graph Neural Networks (GNNs) have achieved enormous success in multiple graph analytical tasks, modern variants mostly rely on the strong inductive bias of homophily. However, real-world networks typically exhibit both homophilic and heterophilic linking patterns, wherein adjacent nodes may share dissimilar attributes and distinct labels. Therefore, GNNs smoothing node proximity holistically may aggregate both task-relevant and irrelevant (even harmful) information, limiting their ability to generalize to heterophilic graphs and potentially causing non-robustness. In this work, we propose a novel edge splitting GNN (ES-GNN) framework to adaptively distinguish between graph edges either relevant or irrelevant to learning tasks. This essentially transfers the original graph into two subgraphs with the same node set but exclusive edge sets dynamically. Given that, information propagation separately on these subgraphs and edge splitting are alternatively conducted, thus disentangling the task-relevant and irrelevant features. Theoretically, we show that our ES-GNN can be regarded as a solution to a disentangled graph denoising problem, which further illustrates our motivations and interprets the improved generalization beyond homophily. Extensive experiments over 11 benchmark and 1 synthetic datasets demonstrate that ES-GNN not only outperforms the state-of-the-arts, but also can be more robust to adversarial graphs and alleviate the over-smoothing problem.

## Pipeline
<p align = "center">
<img src=""/>
</p>
<p align = "left">
Figure 1: 
</p>


## Datasets
<p align = "center">
<img src = "">
</p>
<p align = "left">
Figure 2: 
</p>

## 


## Citation
```
@article{guo2022gnn,
  title={ES-GNN: Generalizing Graph Neural Networks Beyond Homophily with Edge Splitting},
  author={Guo, Jingwei and Huang, Kaizhu and Yi, Xinping and Zhang, Rui},
  journal={arXiv preprint arXiv:2205.13700},
  year={2022}
}
```
