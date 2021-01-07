# PCA-information-access

Author: Nasanbayar Ulzii-Orshikh

Date: 01/06/2021

Clustering nodes in a social network based on their degree of information access is a tool with enormous potential to address the epistemic inequity, rooted from digital space (Beilinson et al. 2020).
However, producing a network representation, called ``information access signatures", that is appropriate for such clustering is computationally expensive, taking at least $O(n^2$) time.
Then, this project explores three representations of the 114th Congressional co-sponsorship network -- information access signatures, adjacency matrix, and representation with node2vec -- using Principal Component Analysis (PCA) to find and experimentally validate a computationally less expensive pipeline of information access clustering.

Project's pipeline and relation to Beilinson et al. 2020:

![project_pipeline.jpeg][project_pipeline.jpeg]

Example input: python3 main_pipelines.py -m info_access -n 2

-m is the mode from {info_access, adjacency, node2vec, all} and -n is the number of PCs from [1,3].

Hypothesis testing results (Fisher Exact tests) can be found in the qualitative_analysis folder. Relevant plots in the plots one.

The full paper manuscript can be provided upon request. E-mail: nulziiorsh@haverford.edu. Thank you! :)

Citation: Hannah C. Beilinson et al., "Clustering via Information Access in a Network", 2020.