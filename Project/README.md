
# Gaussian Processes and Representation Learning

Neil D. Lawrence (2005) Probabilistic non‐linear principal component analysis with Gaussian process latent variable models". The Journal of Machine Learning Research 6, pp. 1783‐1816.
---
Neil D. Lawrence (2004) Gaussian process models for visualisation of high dimensional data. In S. Thrun, L. Saul, and B. Scholkopf, editors, Advances in Neural Information Processing Systems, vol. 16, pp. 329–336, Cambridge, MA, MIT Press.
---
These papers introduce the concept of Gaussian process latent variable model (GP‐LVM) that can be considered as a multiple‐output GP regression model where only the output data are given. In this respect the model implements representation learning with the unobserved inputs treated as latent variables, which are optimised instead of being integrated out. This perspective renders the model as a nonlinear extension of the linear
probabilistic principal component analysis (PPCA). In the other paper GPLVM is then evaluated as an approach to the visualisation of high dimensional data.
---
In the project you could examine in depth and build on the following tasks
---
- reimplement the proposed method by following model development in the paper
including the MAP estimate and its maximization for hyperparameter identification
− demonstrate the use of the model for visualization of new high‐dimensional datasets
− make a comparative evaluation of GPLVM with a kernel PCA algorithm on selected
high‐dimensional datasets
− extend the proposed GPLVM algorithm by devising a variational Bayesian approach
to the marginalization of latent (input) variables, which would optimize the lower
bound on the marginal likelihood wrt hyperparameters
− another option for extension could be model sparsification
− elaborate further on the discussion points in Lawrence (2004).

---
You are requested to develop your own implementations rather than rely on any existing libraries, particularly the code referred to in the papers.
