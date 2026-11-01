/* =====================================================================
   4MLSim module registry: every simulation, its section, level and status.
   status: 'ready' = built; 'planned' = in development (shown greyed out)
   ===================================================================== */
(function () {
  'use strict';
  const MLS = window.MLS;

  MLS.SECTIONS = [
    { id: 'foundations', title: 'Foundations', blurb: 'Fitting, error, complexity and honest evaluation: the ideas every model rests on.' },
    { id: 'supervised', title: 'Supervised Learning', blurb: 'Classifiers that learn a boundary between labelled groups.' },
    { id: 'trees', title: 'Trees & Ensembles', blurb: 'Decision trees and the forests and boosted models built from them.' },
    { id: 'unsupervised', title: 'Unsupervised Learning', blurb: 'Finding groups and structure in data without labels.' },
    { id: 'neural', title: 'Neural Networks', blurb: 'From one neuron to networks that learn by backpropagation and remember sequences.' },
    { id: 'evaluation', title: 'Evaluation & Model Selection', blurb: 'Measuring performance, tuning, and not fooling yourself.' },
    { id: 'rl', title: 'Reinforcement Learning', blurb: 'Agents that learn by trial, error and reward.' },
    { id: 'modern', title: 'Modern AI', blurb: 'How large language models turn text into predictions.' },
    { id: 'tools', title: 'Reference Tools', blurb: 'Choosing an algorithm and looking up terms.' }
  ];

  /* icon glyphs (viewBox 0 0 30 30) */
  const S1 = 'var(--series-1)', S2 = 'var(--series-2)', S3 = 'var(--series-3)', INK = 'var(--chart-ink)', MU = 'var(--chart-muted)';
  const I = {
    gd: `<path d="M3 6 C9 6,10 24,15 24 S21 6,27 6" stroke="${MU}" stroke-width="1.6" fill="none"/><circle cx="6" cy="7.5" r="2" fill="${S2}"/><circle cx="9.5" cy="14" r="2" fill="${S2}"/><circle cx="12.5" cy="21" r="2" fill="${S2}"/><circle cx="15" cy="24" r="2.4" fill="${S3}"/>`,
    line: `<circle cx="6" cy="22" r="2" fill="${S1}"/><circle cx="11" cy="17" r="2" fill="${S1}"/><circle cx="16" cy="18" r="2" fill="${S1}"/><circle cx="21" cy="11" r="2" fill="${S1}"/><circle cx="25" cy="8" r="2" fill="${S1}"/><path d="M3 24 L27 6" stroke="${S2}" stroke-width="2.2" stroke-linecap="round"/>`,
    poly: `<circle cx="5" cy="18" r="2" fill="${S1}"/><circle cx="10" cy="9" r="2" fill="${S1}"/><circle cx="15" cy="14" r="2" fill="${S1}"/><circle cx="20" cy="21" r="2" fill="${S1}"/><circle cx="25" cy="10" r="2" fill="${S1}"/><path d="M3 22 C6 2,10 6,14 14 S20 30,27 6" stroke="${S2}" stroke-width="2" fill="none"/>`,
    target: `<circle cx="15" cy="15" r="11" fill="none" stroke="${MU}" stroke-width="1.6"/><circle cx="15" cy="15" r="6.5" fill="none" stroke="${MU}" stroke-width="1.6"/><circle cx="15" cy="15" r="2.2" fill="${S2}"/><circle cx="9" cy="10" r="1.6" fill="${S1}"/><circle cx="11" cy="8" r="1.6" fill="${S1}"/><circle cx="8" cy="12" r="1.6" fill="${S1}"/>`,
    reg: `<path d="M4 24 L4 6 M4 24 L26 24" stroke="${MU}" stroke-width="1.5"/><path d="M6 8 C12 14,18 18,26 19" stroke="${S1}" stroke-width="2" fill="none"/><path d="M6 12 C12 16,18 20,26 21" stroke="${S2}" stroke-width="2" fill="none" stroke-dasharray="3 2"/>`,
    scale: `<rect x="4" y="18" width="5" height="8" rx="1" fill="${S1}"/><rect x="12" y="6" width="5" height="20" rx="1" fill="${S2}"/><rect x="20" y="14" width="5" height="12" rx="1" fill="${S3}"/>`,
    leak: `<path d="M6 6 H24 V14 Q24 22 15 25 Q6 22 6 14 Z" fill="none" stroke="${INK}" stroke-width="1.6"/><path d="M15 12 Q12 16 15 19 Q18 16 15 12 Z" fill="${S2}"/><path d="M24 10 L28 8" stroke="${S2}" stroke-width="2"/>`,
    tsval: `<path d="M3 20 L8 15 L12 17 L17 10 L21 13 L27 7" fill="none" stroke="${INK}" stroke-width="1.6"/><rect x="3" y="23" width="14" height="4" rx="1" fill="${S1}"/><rect x="18" y="23" width="4" height="4" rx="1" fill="${S2}"/><rect x="23" y="23" width="4" height="4" rx="1" fill="${MU}"/>`,
    split: `<rect x="3" y="9" width="15" height="12" rx="2" fill="${S1}"/><rect x="19" y="9" width="4" height="12" rx="1" fill="${S2}"/><rect x="24" y="9" width="3" height="12" rx="1" fill="${S3}"/>`,
    sigmoid: `<path d="M3 23 C12 23,12 7,27 7" stroke="${S2}" stroke-width="2.4" fill="none"/><circle cx="6" cy="23" r="1.8" fill="${S1}"/><circle cx="9" cy="23" r="1.8" fill="${S1}"/><circle cx="21" cy="7" r="1.8" fill="${S3}"/><circle cx="25" cy="7" r="1.8" fill="${S3}"/>`,
    knn: `<circle cx="15" cy="15" r="9" fill="none" stroke="${MU}" stroke-dasharray="2 2"/><circle cx="15" cy="15" r="2.4" fill="${INK}"/><circle cx="10" cy="11" r="2" fill="${S1}"/><circle cx="20" cy="12" r="2" fill="${S1}"/><rect x="16" y="18" width="4" height="4" fill="${S2}"/><rect x="24" y="3" width="4" height="4" fill="${S2}"/>`,
    bayes: `<path d="M3 24 C8 24,9 8,13 8 S18 24,22 24" stroke="${S1}" stroke-width="2" fill="none"/><path d="M8 24 C13 24,14 10,18 10 S23 24,28 24" stroke="${S2}" stroke-width="2" fill="none"/>`,
    svm: `<path d="M4 26 L26 4" stroke="${INK}" stroke-width="2"/><path d="M4 19 L19 4 M11 26 L26 11" stroke="${MU}" stroke-dasharray="2 2"/><circle cx="8" cy="10" r="2" fill="${S1}"/><circle cx="12" cy="6" r="2" fill="${S1}"/><rect x="18" y="19" width="4" height="4" fill="${S2}"/><rect x="22" y="15" width="4" height="4" fill="${S2}"/>`,
    kernel: `<circle cx="15" cy="15" r="7" fill="none" stroke="${INK}" stroke-width="2"/><circle cx="15" cy="15" r="2" fill="${S1}"/><circle cx="13" cy="17" r="1.6" fill="${S1}"/><rect x="3" y="4" width="3.5" height="3.5" fill="${S2}"/><rect x="24" y="22" width="3.5" height="3.5" fill="${S2}"/><rect x="24" y="5" width="3.5" height="3.5" fill="${S2}"/><rect x="4" y="22" width="3.5" height="3.5" fill="${S2}"/>`,
    tree: `<rect x="11" y="3" width="8" height="5" rx="1.5" fill="${INK}"/><path d="M15 8 L8 14 M15 8 L22 14 M8 19 L5 23 M8 19 L11 23" stroke="${MU}" stroke-width="1.5"/><rect x="4" y="14" width="8" height="5" rx="1.5" fill="${S1}"/><rect x="18" y="14" width="8" height="5" rx="1.5" fill="${S2}"/><circle cx="5" cy="25" r="2" fill="${S1}"/><circle cx="11" cy="25" r="2" fill="${S3}"/>`,
    forest: `<path d="M8 4 L3 14 H13 Z M8 13 L2 24 H14 Z" fill="${S3}"/><path d="M21 6 L16 15 H26 Z M21 14 L15 25 H27 Z" fill="${S1}"/><rect x="7" y="24" width="2" height="3" fill="${MU}"/><rect x="20" y="25" width="2" height="3" fill="${MU}"/>`,
    boost: `<rect x="3" y="20" width="5" height="6" rx="1" fill="${S1}"/><rect x="10" y="14" width="5" height="12" rx="1" fill="${S1}"/><rect x="17" y="9" width="5" height="17" rx="1" fill="${S1}"/><path d="M4 18 L12 11 L19 6 L27 3" stroke="${S2}" stroke-width="2" fill="none"/>`,
    xgb: `<path d="M3 25 L9 25 L9 18 L15 18 L15 12 L21 12 L21 7 L27 7" stroke="${S2}" stroke-width="2.2" fill="none"/><circle cx="6" cy="22" r="1.7" fill="${S1}"/><circle cx="12" cy="17" r="1.7" fill="${S1}"/><circle cx="18" cy="13" r="1.7" fill="${S1}"/><circle cx="24" cy="6" r="1.7" fill="${S1}"/>`,
    kmeans: `<circle cx="8" cy="9" r="1.8" fill="${S1}"/><circle cx="11" cy="7" r="1.8" fill="${S1}"/><circle cx="7" cy="12" r="1.8" fill="${S1}"/><rect x="18" y="18" width="3.4" height="3.4" fill="${S2}"/><rect x="22" y="21" width="3.4" height="3.4" fill="${S2}"/><rect x="21" y="15" width="3.4" height="3.4" fill="${S2}"/><path d="M8 4 v6 M5 7 h6" stroke="${INK}" stroke-width="1.6" transform="translate(1 2)"/><path d="M22 16 v6 M19 19 h6" stroke="${INK}" stroke-width="1.6"/>`,
    dendro: `<path d="M4 26 V18 H10 V26 M7 18 V12 H18 V20 M15 26 V20 H21 V26 M12.5 12 V5 H25 V26" stroke="${S1}" stroke-width="1.8" fill="none"/>`,
    dbscan: `<path d="M4 10 C8 4,14 4,16 8 S24 12,26 8" stroke="${S1}" stroke-width="3" fill="none" stroke-dasharray="0.1 4" stroke-linecap="round"/><path d="M4 22 C8 16,14 16,16 20 S24 24,26 20" stroke="${S2}" stroke-width="3" fill="none" stroke-dasharray="0.1 4" stroke-linecap="round"/><circle cx="25" cy="27" r="1.6" fill="${MU}"/>`,
    gmm: `<ellipse cx="11" cy="12" rx="8" ry="5" transform="rotate(-25 11 12)" fill="none" stroke="${S1}" stroke-width="1.8"/><ellipse cx="20" cy="20" rx="7" ry="4" transform="rotate(30 20 20)" fill="none" stroke="${S2}" stroke-width="1.8"/><circle cx="11" cy="12" r="1.6" fill="${S1}"/><circle cx="20" cy="20" r="1.6" fill="${S2}"/>`,
    pca: `<circle cx="8" cy="21" r="1.7" fill="${S1}"/><circle cx="12" cy="17" r="1.7" fill="${S1}"/><circle cx="15" cy="16" r="1.7" fill="${S1}"/><circle cx="19" cy="11" r="1.7" fill="${S1}"/><circle cx="22" cy="10" r="1.7" fill="${S1}"/><path d="M4 25 L27 5" stroke="${S2}" stroke-width="2"/><path d="M11 9 L19 19" stroke="${S3}" stroke-width="2"/>`,
    dims: `<path d="M5 22 L13 26 L25 20 L17 16 Z" fill="none" stroke="${MU}" stroke-width="1.4"/><path d="M5 22 V10 L17 4 V16 M17 4 L25 8 V20" stroke="${MU}" stroke-width="1.4" fill="none"/><circle cx="12" cy="13" r="1.8" fill="${S1}"/><circle cx="20" cy="12" r="1.8" fill="${S2}"/>`,
    ann: `<path d="M6 9 L15 6 M6 9 L15 15 M6 9 L15 24 M6 21 L15 6 M6 21 L15 15 M6 21 L15 24 M15 6 L24 15 M15 15 L24 15 M15 24 L24 15" stroke="${MU}" stroke-width="1.1"/><circle cx="6" cy="9" r="2.6" fill="${S1}"/><circle cx="6" cy="21" r="2.6" fill="${S1}"/><circle cx="15" cy="6" r="2.6" fill="${S3}"/><circle cx="15" cy="15" r="2.6" fill="${S3}"/><circle cx="15" cy="24" r="2.6" fill="${S3}"/><circle cx="24" cy="15" r="2.6" fill="${S2}"/>`,
    lstm: `<rect x="3" y="10" width="7" height="10" rx="2" fill="${S1}"/><rect x="12" y="10" width="7" height="10" rx="2" fill="${S1}"/><rect x="21" y="10" width="7" height="10" rx="2" fill="${S1}"/><path d="M10 15 h2 M19 15 h2" stroke="${S2}" stroke-width="2"/><path d="M6.5 7 C6.5 3,24.5 3,24.5 7" stroke="${S2}" stroke-width="1.6" fill="none"/>`,
    cm: `<rect x="4" y="4" width="10" height="10" rx="2" fill="${S1}"/><rect x="16" y="4" width="10" height="10" rx="2" fill="${MU}" opacity="0.35"/><rect x="4" y="16" width="10" height="10" rx="2" fill="${MU}" opacity="0.35"/><rect x="16" y="16" width="10" height="10" rx="2" fill="${S1}"/>`,
    roc: `<path d="M4 26 L4 4 M4 26 L26 26" stroke="${MU}" stroke-width="1.4"/><path d="M4 26 L26 4" stroke="${MU}" stroke-dasharray="2 2"/><path d="M4 26 C5 12,10 6,26 4" stroke="${S2}" stroke-width="2.2" fill="none"/>`,
    grid: `${[0, 1, 2, 3].map(i => [0, 1, 2, 3].map(j => `<rect x="${4 + i * 6}" y="${4 + j * 6}" width="5" height="5" rx="1" fill="${S1}" opacity="${(0.2 + 0.8 * Math.exp(-((i - 2) ** 2 + (j - 1) ** 2) / 3)).toFixed(2)}"/>`).join('')).join('')}`,
    snoop: `${[6, 10, 14, 18, 22, 26].map((x, k) => `<rect x="${x - 2}" y="${[14, 18, 11, 16, 6, 13][k]}" width="3" height="${26 - [14, 18, 11, 16, 6, 13][k]}" rx="1" fill="${k === 4 ? S2 : S1}"/>`).join('')}`,
    explain: `<rect x="4" y="5" width="20" height="4" rx="1" fill="${S1}"/><rect x="4" y="12" width="14" height="4" rx="1" fill="${S1}"/><rect x="4" y="19" width="8" height="4" rx="1" fill="${S2}"/>`,
    gridworld: `<rect x="3" y="3" width="24" height="24" rx="2" fill="none" stroke="${MU}"/><path d="M11 3 V27 M19 3 V27 M3 11 H27 M3 19 H27" stroke="${MU}" stroke-width="0.8"/><rect x="20" y="4" width="6" height="6" fill="${S3}"/><rect x="12" y="12" width="6" height="6" fill="${S2}"/><path d="M6 23 L6 15 L14 7" stroke="${S1}" stroke-width="2" fill="none"/>`,
    llm: `<rect x="3" y="6" width="7" height="5" rx="1.5" fill="${S1}"/><rect x="12" y="6" width="9" height="5" rx="1.5" fill="${S1}"/><rect x="3" y="15" width="24" height="3" rx="1" fill="${S2}"/><rect x="3" y="20" width="15" height="3" rx="1" fill="${S2}" opacity="0.7"/><rect x="3" y="25" width="7" height="3" rx="1" fill="${S2}" opacity="0.45"/>`,
    guide: `<path d="M15 3 L15 9 M15 9 L7 15 M15 9 L23 15" stroke="${MU}" stroke-width="1.6"/><circle cx="15" cy="5" r="3" fill="${INK}"/><rect x="3" y="15" width="8" height="6" rx="1.5" fill="${S1}"/><rect x="19" y="15" width="8" height="6" rx="1.5" fill="${S2}"/><path d="M7 21 V26 M23 21 V26" stroke="${MU}" stroke-width="1.6"/>`,
    glossary: `<rect x="6" y="3" width="18" height="24" rx="2" fill="none" stroke="${INK}" stroke-width="1.6"/><path d="M10 9 H20 M10 14 H20 M10 19 H16" stroke="${S1}" stroke-width="2"/>`
  };
  MLS.moduleIcon = key => '<svg viewBox="0 0 30 30" aria-hidden="true">' + (I[key] || I.line) + '</svg>';

  /* level: Intro | Intermediate | Advanced */
  MLS.MODULES = [
    { id: 'linear-regression', section: 'foundations', level: 'Intro', icon: 'line', status: 'ready', title: 'Linear Regression: Econometrics & ML', blurb: 'One equation, two readings: coefficients, standard errors and diagnostic tests next to out-of-sample prediction error.' },
    { id: 'polynomial-overfitting', section: 'foundations', level: 'Intro', icon: 'poly', status: 'ready', title: 'Polynomial Overfitting', blurb: 'Raise the polynomial degree and compare training and test error to see overfitting happen.' },
    { id: 'gradient-descent', section: 'foundations', level: 'Intro', icon: 'gd', status: 'ready', title: 'Gradient Descent', blurb: 'How machine learning models are trained: step downhill on the loss surface. Learning rate, scaling, local minima.' },
    { id: 'bias-variance', section: 'foundations', level: 'Intermediate', icon: 'target', status: 'ready', title: 'Bias-Variance Tradeoff', blurb: 'Refit a model on many fresh samples; measure bias², variance and noise as complexity grows.' },
    { id: 'regularization', section: 'foundations', level: 'Intermediate', icon: 'reg', status: 'ready', title: 'Regularization: Ridge, Lasso, Elastic Net', blurb: 'Shrink coefficients with a penalty λ and follow their paths; see Lasso set some exactly to zero.' },
    { id: 'feature-scaling', section: 'foundations', level: 'Intro', icon: 'scale', status: 'ready', title: 'Feature Scaling', blurb: 'Income in euros next to age in years: how unscaled features distort distances and slow down learning.' },
    { id: 'cross-validation', section: 'foundations', level: 'Intro', icon: 'split', status: 'ready', title: 'Cross-Validation', blurb: 'Training set, validation folds and test (hold-out) set, step by step: fit, validate, average, choose, refit, test once.' },
    { id: 'data-leakage', section: 'foundations', level: 'Intermediate', icon: 'leak', status: 'ready', title: 'Data Leakage', blurb: 'Three ways information from outside the training data sneaks in: the model looks better than it is, but on new data it does no better.' },
    { id: 'ts-validation', section: 'foundations', level: 'Intermediate', icon: 'tsval', status: 'ready', title: 'Time-Series Validation', blurb: 'Time-series split, expanding and rolling windows, the gap for longer horizons, one final test, and why shuffled folds fail.' },

    { id: 'logistic-regression', section: 'supervised', level: 'Intro', icon: 'sigmoid', status: 'ready', title: 'Logistic Regression', blurb: 'Turn a score into a probability with the sigmoid, estimate by maximum likelihood, and pick a threshold.' },
    { id: 'knn', section: 'supervised', level: 'Intro', icon: 'knn', status: 'ready', title: 'K-Nearest Neighbors', blurb: 'Classify by a vote of the closest points; see the boundary change with k and the distance measure.' },
    { id: 'naive-bayes', section: 'supervised', level: 'Intermediate', icon: 'bayes', status: 'ready', title: 'Naive Bayes', blurb: 'Combine class priors and feature likelihoods with Bayes rule; why the naive assumption still works.' },
    { id: 'svm-linear', section: 'supervised', level: 'Intermediate', icon: 'svm', status: 'ready', title: 'Support Vector Machine (Linear)', blurb: 'The maximum-margin line, support vectors, and the soft-margin cost C.' },
    { id: 'svm-kernel', section: 'supervised', level: 'Advanced', icon: 'kernel', status: 'ready', title: 'Support Vector Machine (Kernel Trick)', blurb: 'RBF and polynomial kernels bend the boundary; see the data lifted into a higher dimension.' },

    { id: 'decision-tree', section: 'trees', level: 'Intro', icon: 'tree', status: 'ready', title: 'Decision Tree', blurb: 'Grow a tree split by split with Gini or entropy, for classification and regression.' },
    { id: 'random-forest', section: 'trees', level: 'Intermediate', icon: 'forest', status: 'ready', title: 'Random Forest', blurb: 'Bootstrap samples plus random features: many noisy trees vote into one stable model.' },
    { id: 'gradient-boosting', section: 'trees', level: 'Intermediate', icon: 'boost', status: 'ready', title: 'Gradient Boosting', blurb: 'Add small trees one at a time, each fitted to the remaining errors, with a learning rate.' },
    { id: 'xgboost', section: 'trees', level: 'Advanced', icon: 'xgb', status: 'ready', title: 'XGBoost', blurb: 'Gradient and Hessian statistics, split gain, regularization λ and γ, and shrinkage.' },

    { id: 'k-means', section: 'unsupervised', level: 'Intro', icon: 'kmeans', status: 'ready', title: 'K-Means Clustering', blurb: 'Assign, update, repeat: watch centroids converge; k-means++, the elbow and silhouette.' },
    { id: 'hierarchical', section: 'unsupervised', level: 'Intro', icon: 'dendro', status: 'ready', title: 'Hierarchical Clustering', blurb: 'Merge the closest clusters step by step and cut the dendrogram; four linkage methods.' },
    { id: 'dbscan', section: 'unsupervised', level: 'Intermediate', icon: 'dbscan', status: 'ready', title: 'DBSCAN', blurb: 'Density-based clusters of any shape, with core, border and noise points.' },
    { id: 'gmm-em', section: 'unsupervised', level: 'Advanced', icon: 'gmm', status: 'ready', title: 'Gaussian Mixtures & EM', blurb: 'Soft clustering with ellipses; the E-step and M-step one at a time.' },
    { id: 'pca', section: 'unsupervised', level: 'Intermediate', icon: 'pca', status: 'ready', title: 'Principal Component Analysis', blurb: 'Rotate to the directions of most variance, project, and read the scree plot.' },
    { id: 'curse-of-dimensionality', section: 'unsupervised', level: 'Intermediate', icon: 'dims', status: 'ready', title: 'Curse of Dimensionality', blurb: 'As features are added, distances bunch together and neighbours stop being near.' },

    { id: 'ann', section: 'neural', level: 'Intermediate', icon: 'ann', status: 'ready', title: 'Neural Networks: Forward Pass & Backpropagation', blurb: 'Trace every weighted sum, then train the network and watch the boundary and the gradients.' },
    { id: 'lstm', section: 'neural', level: 'Advanced', icon: 'lstm', status: 'ready', title: 'LSTM for Time Series', blurb: 'An unrolled recurrent network with forget, input and output gates on an economic series.' },

    { id: 'confusion-matrix', section: 'evaluation', level: 'Intro', icon: 'cm', status: 'ready', title: 'Confusion Matrix & Metrics', blurb: 'Move the threshold and see accuracy, precision, recall, specificity and F1 respond.' },
    { id: 'roc-pr-calibration', section: 'evaluation', level: 'Intermediate', icon: 'roc', status: 'ready', title: 'ROC, Precision-Recall & Calibration', blurb: 'ROC and PR curves, AUC, class imbalance, cost-based thresholds and calibrated probabilities.' },
    { id: 'hyperparameter-tuning', section: 'evaluation', level: 'Intermediate', icon: 'grid', status: 'ready', title: 'Hyperparameter Tuning', blurb: 'Grid search versus random search on a validation-score landscape.' },
    { id: 'data-snooping', section: 'evaluation', level: 'Advanced', icon: 'snoop', status: 'ready', title: 'Data Snooping & Multiple Testing', blurb: 'Why the best of many models can be pure luck; Reality Check, SPA and the Model Confidence Set.' },
    { id: 'explainability', section: 'evaluation', level: 'Advanced', icon: 'explain', status: 'ready', title: 'Explainability: Importance, PDP & SHAP', blurb: 'Which features drive a prediction: permutation importance, partial dependence and SHAP values.' },

    { id: 'q-learning', section: 'rl', level: 'Intermediate', icon: 'gridworld', status: 'ready', title: 'Q-Learning in a Gridworld', blurb: 'An agent learns a policy by trial and error; Q-value heatmap and policy arrows.' },

    { id: 'llm-intuition', section: 'modern', level: 'Intermediate', icon: 'llm', status: 'ready', title: 'How LLMs Predict Text', blurb: 'Next-token probabilities, temperature, embeddings and attention on a short sentence.' },

    { id: 'algorithm-guide', section: 'tools', level: 'Intro', icon: 'guide', status: 'ready', tool: true, title: 'Which Algorithm? Guide', blurb: 'A decision guide and a side-by-side comparison of every method in 4MLSim.' },
    { id: 'glossary', section: 'tools', level: 'Intro', icon: 'glossary', status: 'ready', tool: true, title: 'Glossary', blurb: 'Searchable definitions of the terms used across the modules.' }
  ];
  MLS.moduleMeta = function (id) {
    const m = MLS.MODULES.find(x => x.id === id);
    if (!m) return null;
    const s = MLS.SECTIONS.find(x => x.id === m.section);
    return Object.assign({}, m, { sectionTitle: s ? s.title : '' });
  };
})();
