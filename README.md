# 4MLSim: Machine Learning Simulation Lab

Interactive simulations for teaching machine learning, by **Professor Periklis Gogas**
(EMEF Lab, Department of Economics, Democritus University of Thrace).

Open the app: **https://perrygogas.github.io/4MLSim/**  
Click **Install 4MLSim** in the header (Chrome or Edge) to use it as an app; after the first visit
every module also works offline. A single-file version for computers without internet:
`4MLSim-offline.html`.

## The modules

33 simulations in 8 sections, and 2 reference tools. Each module has a Simulation, a Lesson (three levels of mathematics), Challenges and an "On real data" tab.

**Foundations**

- [Linear Regression: Econometrics & ML](https://perrygogas.github.io/4MLSim/modules/linear-regression.html) (Intro): One equation, two readings: coefficients, standard errors and diagnostic tests next to out-of-sample prediction error.
- [Polynomial Overfitting](https://perrygogas.github.io/4MLSim/modules/polynomial-overfitting.html) (Intro): Raise the polynomial degree and compare training and test error to see overfitting happen.
- [Gradient Descent](https://perrygogas.github.io/4MLSim/modules/gradient-descent.html) (Intro): How machine learning models are trained: step downhill on the loss surface. Learning rate, scaling, local minima.
- [Bias-Variance Tradeoff](https://perrygogas.github.io/4MLSim/modules/bias-variance.html) (Intermediate): Refit a model on many fresh samples; measure bias², variance and noise as complexity grows.
- [Regularization: Ridge, Lasso, Elastic Net](https://perrygogas.github.io/4MLSim/modules/regularization.html) (Intermediate): Shrink coefficients with a penalty λ and follow their paths; see Lasso set some exactly to zero.
- [Feature Scaling](https://perrygogas.github.io/4MLSim/modules/feature-scaling.html) (Intro): Income in euros next to age in years: how unscaled features distort distances and slow down learning.
- [Cross-Validation](https://perrygogas.github.io/4MLSim/modules/cross-validation.html) (Intro): Training set, validation folds and test (hold-out) set, step by step: fit, validate, average, choose, refit, test once.
- [Data Leakage](https://perrygogas.github.io/4MLSim/modules/data-leakage.html) (Intermediate): Three ways information from outside the training data sneaks in: the model looks better than it is, but on new data it does no better.
- [Time-Series Validation](https://perrygogas.github.io/4MLSim/modules/ts-validation.html) (Intermediate): Time-series split, expanding and rolling windows, the gap for longer horizons, one final test, and why shuffled folds fail.

**Supervised Learning**

- [Logistic Regression](https://perrygogas.github.io/4MLSim/modules/logistic-regression.html) (Intro): Turn a score into a probability with the sigmoid, estimate by maximum likelihood, and pick a threshold.
- [K-Nearest Neighbors](https://perrygogas.github.io/4MLSim/modules/knn.html) (Intro): Classify by a vote of the closest points; see the boundary change with k and the distance measure.
- [Naive Bayes](https://perrygogas.github.io/4MLSim/modules/naive-bayes.html) (Intermediate): Combine class priors and feature likelihoods with Bayes rule; why the naive assumption still works.
- [Support Vector Machine (Linear)](https://perrygogas.github.io/4MLSim/modules/svm-linear.html) (Intermediate): The maximum-margin line, support vectors, and the soft-margin cost C.
- [Support Vector Machine (Kernel Trick)](https://perrygogas.github.io/4MLSim/modules/svm-kernel.html) (Advanced): RBF and polynomial kernels bend the boundary; see the data lifted into a higher dimension.

**Trees & Ensembles**

- [Decision Tree](https://perrygogas.github.io/4MLSim/modules/decision-tree.html) (Intro): Grow a tree split by split with Gini or entropy, for classification and regression.
- [Random Forest](https://perrygogas.github.io/4MLSim/modules/random-forest.html) (Intermediate): Bootstrap samples plus random features: many noisy trees vote into one stable model.
- [Gradient Boosting](https://perrygogas.github.io/4MLSim/modules/gradient-boosting.html) (Intermediate): Add small trees one at a time, each fitted to the remaining errors, with a learning rate.
- [XGBoost](https://perrygogas.github.io/4MLSim/modules/xgboost.html) (Advanced): Gradient and Hessian statistics, split gain, regularization λ and γ, and shrinkage.

**Unsupervised Learning**

- [K-Means Clustering](https://perrygogas.github.io/4MLSim/modules/k-means.html) (Intro): Assign, update, repeat: watch centroids converge; k-means++, the elbow and silhouette.
- [Hierarchical Clustering](https://perrygogas.github.io/4MLSim/modules/hierarchical.html) (Intro): Merge the closest clusters step by step and cut the dendrogram; four linkage methods.
- [DBSCAN](https://perrygogas.github.io/4MLSim/modules/dbscan.html) (Intermediate): Density-based clusters of any shape, with core, border and noise points.
- [Gaussian Mixtures & EM](https://perrygogas.github.io/4MLSim/modules/gmm-em.html) (Advanced): Soft clustering with ellipses; the E-step and M-step one at a time.
- [Principal Component Analysis](https://perrygogas.github.io/4MLSim/modules/pca.html) (Intermediate): Rotate to the directions of most variance, project, and read the scree plot.
- [Curse of Dimensionality](https://perrygogas.github.io/4MLSim/modules/curse-of-dimensionality.html) (Intermediate): As features are added, distances bunch together and neighbours stop being near.

**Neural Networks**

- [Neural Networks: Forward Pass & Backpropagation](https://perrygogas.github.io/4MLSim/modules/ann.html) (Intermediate): Trace every weighted sum, then train the network and watch the boundary and the gradients.
- [LSTM for Time Series](https://perrygogas.github.io/4MLSim/modules/lstm.html) (Advanced): An unrolled recurrent network with forget, input and output gates on an economic series.

**Evaluation & Model Selection**

- [Confusion Matrix & Metrics](https://perrygogas.github.io/4MLSim/modules/confusion-matrix.html) (Intro): Move the threshold and see accuracy, precision, recall, specificity and F1 respond.
- [ROC, Precision-Recall & Calibration](https://perrygogas.github.io/4MLSim/modules/roc-pr-calibration.html) (Intermediate): ROC and PR curves, AUC, class imbalance, cost-based thresholds and calibrated probabilities.
- [Hyperparameter Tuning](https://perrygogas.github.io/4MLSim/modules/hyperparameter-tuning.html) (Intermediate): Grid search versus random search on a validation-score landscape.
- [Data Snooping & Multiple Testing](https://perrygogas.github.io/4MLSim/modules/data-snooping.html) (Advanced): Why the best of many models can be pure luck; Reality Check, SPA and the Model Confidence Set.
- [Explainability: Importance, PDP & SHAP](https://perrygogas.github.io/4MLSim/modules/explainability.html) (Advanced): Which features drive a prediction: permutation importance, partial dependence and SHAP values.

**Reinforcement Learning**

- [Q-Learning in a Gridworld](https://perrygogas.github.io/4MLSim/modules/q-learning.html) (Intermediate): An agent learns a policy by trial and error; Q-value heatmap and policy arrows.

**Modern AI**

- [How LLMs Predict Text](https://perrygogas.github.io/4MLSim/modules/llm-intuition.html) (Intermediate): Next-token probabilities, temperature, embeddings and attention on a short sentence.

**Reference Tools**

- [Which Algorithm? Guide](https://perrygogas.github.io/4MLSim/modules/algorithm-guide.html) (Intro): A decision guide and a side-by-side comparison of every method in 4MLSim.
- [Glossary](https://perrygogas.github.io/4MLSim/modules/glossary.html) (Intro): Searchable definitions of the terms used across the modules.

## What each module offers

- Economic and synthetic datasets with a fixed random seed, your own clicked points, or a CSV / Excel file.
- Three math levels: Intuition, Formulas, Deep dive.
- Step-by-step runs with highlighted pseudocode for iterative algorithms.
- "Predict first" questions and checked challenges.
- Share links that open a module with the same data, seed and settings.
- Projector mode, PNG / SVG figure export, chart data to Excel.

## Repository layout

```
index.html              home menu
about.html              version, citation, license
modules/*.html          one page per simulation
assets/css, assets/js   shared style and engine (core.js, plot.js, modules.js),
                        verified algorithms (core-ml.js) and module helpers (*-ui.js)
assets/vendor           Temml (math), SheetJS (Excel/CSV)
assets/fonts, img       self-hosted fonts, icons, EMEF logo
manifest.json, sw.js    installable app and offline support
tools/build.py          regenerates sw.js and 4MLSim-offline.html
```

After changing any file run `python3 tools/build.py` before uploading, so the offline cache
and the single-file version include the change.

## Publishing on GitHub Pages

1. Upload all files and folders to the repository `PerryGogas/4MLSim` (keep the folder structure).
2. Settings > Pages > Source: *Deploy from a branch*, branch `main`, folder `/ (root)`.
3. The app appears at `https://perrygogas.github.io/4MLSim/`.
4. To announce a new version to users, create a GitHub Release with a tag such as `v1.0.1`
   (the app compares it with its own version and shows a notice).

## Checked against standard software

The algorithms in `assets/js/core-ml.js` were checked against statsmodels, scikit-learn, SciPy, xgboost and arch
(regression and logit coefficients, naive Bayes, KNN, trees, SVM, gradient boosting, XGBoost, k-means, silhouette,
DBSCAN, hierarchical clustering, Gaussian mixtures, PCA, Reality Check, Model Confidence Set); neural network and LSTM
gradients against numerical derivatives; Q-learning against value iteration.

## How to cite

Gogas, P. (2026). *4MLSim: Machine Learning Simulation Lab* (Version 1.0.0) [Computer software].
EMEF Lab, Department of Economics, Democritus University of Thrace. https://github.com/PerryGogas/4MLSim

## License

MIT License (see `LICENSE`). Third-party components keep their own licenses.
