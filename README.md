# ES-GNN: Generalizing Graph Neural Networks Beyond Homophily with Edge Splitting

## Abstract
While Graph Neural Networks (GNNs) have achieved enormous success in multiple graph analytical tasks, modern variants mostly rely on the strong inductive bias of homophily. However, real-world networks typically exhibit both homophilic and heterophilic linking patterns, wherein adjacent nodes may share dissimilar attributes and distinct labels. Therefore, GNNs smoothing node proximity holistically may aggregate both task-relevant and irrelevant (even harmful) information, limiting their ability to generalize to heterophilic graphs and potentially causing non-robustness. In this work, we propose a novel edge splitting GNN (ES-GNN) framework to adaptively distinguish between graph edges either relevant or irrelevant to learning tasks. This essentially transfers the original graph into two subgraphs with the same node set but exclusive edge sets dynamically. Given that, information propagation separately on these subgraphs and edge splitting are alternatively conducted, thus disentangling the task-relevant and irrelevant features. Theoretically, we show that our ES-GNN can be regarded as a solution to a disentangled graph denoising problem, which further illustrates our motivations and interprets the improved generalization beyond homophily. Extensive experiments over 11 benchmark and 1 synthetic datasets demonstrate that ES-GNN not only outperforms the state-of-the-arts, but also can be more robust to adversarial graphs and alleviate the over-smoothing problem.

## Pipeline
<p align = "center">
<img src="https://github.com/jingweio/ES-GNN/blob/main/esgnn_pipline.png"/>
</p>
<p align = "left">
Figure 1: Illustration of our ES-GNN framework where $\mathbf{A}$ and $\mathbf{X}$ denote the adjacency matrix and feature matrix of nodes, respectively. First, $\mathbf{X}$ is projected onto different latent subspaces via different channels \textit{R} and \textit{IR}. An edge splitting is then performed to divide the original graph edges into two exclusive sets. After that, the node information can be aggregated individually and separately on different edge sets to produce disentangled representations, which are further utilized to make an more accurate edge splitting in the next layer. The task-relevant representation $\mathbf{Z}_R^{'}$ is reasonably granted for prediction. Meanwhile, an Irrelevant Consistency Regularization (ICR) is developed to further reduce the potential task-harmful information from the final predictive target.
</p>

<p align = "center">
<img src="https://github.com/jingweio/ES-GNN/blob/main/toy_example.png" width="800">
</p>
<p align = "left">
Figure 2: A toy example to show differences between conventional GNNs and our ES-GNN in aggregating node features. Conventional GNNs with local smoothness tend to produce non-discriminative representations on heterophilic graphs, while our ES-GNN is able to disentangle and exclude the task-harmful features from the final predictive target.
</p>


## Real Datasets
<p align = "center">
<img src = "https://github.com/jingweio/ES-GNN/blob/main/real_datasets.png">
</p>
<p align = "left">
Table 1: Statistics of real-world datasets, where $\mathcal{H}$ and $\hat{\mathcal{H}}$ (considering class-imbalance problem) provide indexes of graph homophily ratio. It can be observed that, despite the relative high homophily level measured by $\mathcal{H}$ = 0.632, the Twitch-DE dataset with class-imbalance problem is essentially a heterophilic graph as suggested by $\hat{\mathcal{H}}$ = 0.139. For Polblogs dataset, since node features are not provided, we directly use the rows of the adjacency matrix.
</p>

## Synthetic Datasets
<p align = "center">
<img src = "https://github.com/jingweio/ES-GNN/blob/main/syn_datasets.png"  width="800">
</p>
<p align = "left">
Figure 3: 
</p>

## Citation
```
@article{guo2022gnn,
  title={ES-GNN: Generalizing Graph Neural Networks Beyond Homophily with Edge Splitting},
  author={Guo, Jingwei and Huang, Kaizhu and Yi, Xinping and Zhang, Rui},
  journal={arXiv preprint arXiv:2205.13700},
  year={2022}
}
```
