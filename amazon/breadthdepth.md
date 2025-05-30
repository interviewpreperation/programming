# Amazon Breadth and Depth

## Important Literure

 - [DL questions](https://github.com/youssefHosni/Data-Science-Interview-Questions-Answers/blob/main/Deep%20Learning%20Questions%20%26%20Answers%20for%20Data%20Scientists.md)
 - [ML questions](https://github.com/youssefHosni/Data-Science-Interview-Questions-Answers/blob/main/Machine%20Learning%20Interview%20Questions%20%26%20Answers%20for%20Data%20Scientists.md)
 - [Aman's AI Journal - Interview questions](https://aman.ai/primers/ai/interview)


## Postion I apply for

https://www.amazon.jobs/en/jobs/2947006/applied-scientist-ii-artificial-general-intelligence 
- Experience with popular deep learning modelling tools and workflows such as MxNet, TensorFlow, R, scikit-learn, Spark MLLib, numpy, and SciPy.
- Experience with deep learning modelling techniques including CNNs, RNNs, GANs, VAEs, and Transformers.
- Experience with conducting research in a corporate setting.
- Experience in scientific publications at top-tier peer-reviewed conferences or journals.
- Highly effective verbal and written communication skills with both non-technical and technical audiences.
- Experience with conducting applied research in a corporate setting.
- Experience in building deep learning models for business applications.

Explain:
 - [MxNet](https://aws.amazon.com/mxnet)
    - Deep learning workloads can be distributed across multiple GPUs with near-linear scalability, which means that extremely large projects can be handled in less time.
    - As of [September 2023](https://en.wikipedia.org/wiki/Apache_MXNet), it is no longer actively developed.
 - [Spark MLLib](https://aws.amazon.com/what-is/apache-spark/)
    - Machine Learning models can be trained by data scientists with R or Python on any Hadoop data source, saved using MLlib.
    - The algorithms include the ability to do classification, regression, clustering, collaborative filtering, and pattern mining.
 - CNN
 - RNN
 - GANs
 - VAE
 - Transformers

PEFT
https://aman.ai/primers/ai/parameter-efficient-fine-tuning 


## Training and Optimization
- Adaptive gradient approaches, Regularization and overfitting, loss functions
- Bayesian v/s maximum likelihood estimation,
 dealing with class imbalance, K-fold cross validation, bias, and variance

 
## [Evaluation metrics](https://aman.ai/primers/ai/evaluation-metrics/)  
- Accuracy
- Precision, Recall
- Area under ROC
- R-squared: R-squared, also known as the coefficient of determination, is a statistical measure used in regression analysis to assess the goodness of fit of a model. Essentially, it indicates the proportion of the variance in the dependent variable that is predictable from the independent variable(s).
- Mean average precision (MAP)
- Mean reciprocal rank
- Equal Error rate: For biometric systems - authentication.. Binary. The Equal Error Rate is the point where the False Acceptance Rate (FAR) and the False Rejection Rate (FRR) are equal. In other words, it is the operating point where the system makes an equal number of errors in falsely accepting legitimate users (FAR) and falsely rejecting authorized users (FRR).
- Lower: better performance
- A/B testing fundamentals: https://www.youtube.com/watch?v=2sWVLMVQsu0 

## Supervised Learning
- Linear & Logistic regression
- Naive Bayes classifier
- Bagging & Boosting
- K-nearest neighbors
- Trees
- Neural Networks
- Support Vector Machines 
- Random Forests, 
- Gradient Boosted trees, kernel methods, 
- Stochastic Gradient Descent (SGD), 
- Sequence Modeling, 
- Bayesian linear regression, 
- Gaussian Processes, 
- Concepts of overfitting and underfitting, Regularization and evaluation metrics for classification and regression problems

## Unsupervised Learning
### Clustering algorithms, 
- k-Means clustering, 
- Anomaly detection, 
- Markov methods, 
- DBSCAN, 
- Self-organizing maps, 
- Deep Belief Nets, 
- Expectation Maximization (EM), 
- Gaussian Mixture Models (GMM) and 
- Evaluation metrics for clustering problems 

### Probabilistic graphical models
- Bayesian Network, 
- Markov Networks, 
- Variational inference, 
- Markov chain, 
- Monte Carlo methods, 
- Latent Dirichlet Allocation (LDA), 
- Inference methods such as Belief Propagation, 
- Gibbs Sampling 

### Dimensionality reduction
- Auto encoders, 
- t-SNE, 
- Principal Component Analysis (PCA), 
- Singular Value Decomposition (SVD), 
- Spectral Clustering and 
- Matrix Factorization 


## Sequential models
- Hidden Markov model (HMM), 
- Conditional random fields (CRF), 
- Recurrent Neural Network (RNN), 
- Natural Language processing applications such as Named Entity Recognition (NER) and 
- Parts of Speech (POS) tagging 

## Reinforcement Learning
- State–action–reward–state–action (SARSA), 
- explore-exploit techniques, 
- multi-armed bandits 
- epsilon greed
- UCB, 
- Thompson Sampling
- Q-learning, and 
- Deep Q-Networks (DQNs)
- Applied to domains such as retail, Speech, NLP, Vision, robotics, etc. 

https://rlhfbook.com

## Deep Neural Networks / Deep Learning 
- Feed forward Neural Networks 
- Convolutional Neural Networks 
- Backpropagation
- Recurrent Neural Networks (RNNs)
- Long Short Term Memory (LSTM) networks 
- GAN
- Attention
- Dropout
- Vanishing gradient
- Activation Functions 

## Natural Language Processing
- Statistical Language Modelling
- Latent Dirichlet allocation (LDA)
- Named Entity Recognition (NER)
- Word Embedding
- Word2Vec
- Sentiment Analysis
- BERT
- ULMFiT 

Image and Computer Vision 
- Object Detection
- Image recognition
- Pattern recognition
- FaceNet 
- CNN
- YOLO


## Other questions 
### How do you interpret logistic regression?
Logistic regression interprets the relationship between predictor variables and the probability of a binary outcome. It does this by modeling the relationship as a log-odds (log of the odds of success). This log-odds is then transformed back to a probability using the sigmoid function, which outputs a value between 0 and 1.

### How does dropout work?
Randomly setting a fraction of neurons to zero during training. 
Forces the remaining neurons to learn more robust and generalized features, as they are no longer reliant on specific co-adapted neurons.

### What is L1 vs. L2 regularization?
L1 (lasso) → sparsity 
L2 (ridge) → smaller non-zero coefficients
What is the difference between bagging and boosting?
Explain in detail how a 1D CNN works.
https://medium.com/@abhishekjainindore24/understanding-the-1d-convolutional-layer-in-deep-learning-7a4cb994c981 
https://imperialcollegelondon.github.io/ReCoDE-AIForPatents/5_Convolutional_1D_Network_Classification
Describe a case where you have solved an ambiguous business problem using machine learning.

### Having a categorical variable with thousands of distinct values, how would you encode it?
When dealing with a categorical variable with thousands of distinct values (e.g., ZIP codes, product IDs, user IDs), standard encoding methods like one-hot encoding become impractical. Here are effective strategies, ranked by typical preference:
1. Target Encoding (Mean Encoding)
What: Replace each category with the mean of the target variable for that category (e.g., average purchase price per user ID).
Pros:
Preserves information about the target.
Keeps dimensionality low (single column).
Cons:
Risk of overfitting (use smoothing or cross-validation).
May leak target information if not careful.
Improvements:
Add smoothing: (mean_target * n_samples + global_mean * alpha) / (n_samples + alpha).
Use cross-fold target encoding to prevent data leakage.
2. Frequency Encoding
What: Replace categories with their occurrence frequencies (e.g., how often a ZIP code appears in the dataset).
Pros:
Simple, no risk of target leakage.
Works well for tree-based models (e.g., XGBoost).
Cons:
Loses relationship to the target.
Collisions possible (different categories with same frequency).
3. Hash Encoding
What: Use a hash function (e.g., hash(value) mod N) to map categories to a fixed number of bins (e.g., 64 or 256).
Pros:
Constant dimensionality (good for high-cardinality).
No need to store a mapping table.
Cons:
Collisions (multiple categories mapped to same bin).
Hard to interpret.
4. Embedding (for Neural Networks)
What: Train an embedding layer (e.g., via PyTorch/TensorFlow) to map categories to a dense vector (like word embeddings in NLP).
Pros:
Captures complex relationships.
Scalable to millions of categories.
Cons:
Requires a neural network.
Needs sufficient data to train.
5. Rare Value Grouping
What: Group infrequent categories into "Other" or clusters (e.g., ZIP codes by region).
Pros: Reduces noise from rare values.
Cons: Loses granularity.
6. Leave-One-Out Encoding
What: Like target encoding, but for each row, compute the mean target excluding the current row.
Pros: Reduces target leakage.
Cons: Computationally expensive.
When to Use What?
Tree-based models (XGBoost, Random Forest): Frequency or target encoding.
Linear models (Logistic Regression): Target encoding (with smoothing).
Neural Networks: Embeddings or hash encoding.
Low-memory environments: Hash encoding.


Key Considerations
- Avoid one-hot encoding (creates thousands of columns).
- Monitor for overfitting (especially with target encoding).
- Benchmark multiple methods on validation data.


### How do you manage an unbalanced data set?
answered

### What is lstm? Why use lstm? How was lstm used in your experience?


### What did you use to remove multicollinearity? Explain what values of VIF you used.
### Explain different time series analysis models. What are some time series models  other than Arima?
### How does a neural network with one layer and one input and output compare to a  logistic regression?
### How do you evaluate Gen AI models?
### Explain overfitting, When does it happen? How can you avoid it?
### How do you handle overfitting and underfitting in machine learning models?
### What is random forest, and what is C5?
### Random Forest is an ensemble method that combines the predictions of multiple decision trees to improve accuracy, while C5.0 is a single decision tree algorithm that focuses on building a refined and accurate model
### What would a neural network without an activation function look like?
### Simple linear regression model
### How do you perform anomaly detection?
### Anomaly detection identifies unusual patterns or data points that deviate significantly from expected behavior or normal patterns in a dataset. This can be ### achieved through various methods, including statistical techniques, machine learning algorithms, and visual inspection.
### Describe a scenario where collecting more data may not be useful.
### Explain the difference between grid search cv and random search cv.
https://aman.ai/primers/ai/hyperparameter-tuning
In the paper Random Search for Hyper-Parameter Optimization by Bergstra and Bengio, the authors show empirically and theoretically that random search is more efficient for parameter optimization than grid search. 
If there is a defective/unsafe product on Amazon, how would you identify it?
Explain transformer architecture.
###  Build a RAG system. (Book AI Systems 256)
https://aws.amazon.com/what-is/retrieval-augmented-generation 
https://huggingface.co/blog/ngxson/make-your-own-rag 
https://skphd.medium.com/rag-system-design-interview-questions-and-answers-6c0b2865062e  
RAG is a technique that enhances a model’s generation by retrieving the relevant information form external memory sources. An external memory source can be an internal database, a user’s chat sessions, or the internet.
With RAG, only the information most relevant to the query, as determined by the retriever, is retrieved and input into the model. Lewis et al. found that having access to the relevant information can help the model generate more detailed responses while reducing hallucinations. (Meta paper Petroni et al.)

### Trade-off: reduces input tokens vs model performance
### Describe the differences between Adam vs SGD.
### Discuss implementation and fine-tuning of transformer architectures for personalized content or conversational AI, e.g., pretraining on domain-specific datasets for Alexa Skills.


## Amazon data scientist interview questions: statistics
### What is p-value?
### What is the maximum likelihood of getting k heads when you tossed a coin n times?  Write down the mathematics behind it.
### There are 4 red balls and 2 blue balls, what's the probability of them not being the same in the 2 picks?
### How would you explain hypothesis testing for a newbie?
### What is cross-validation?
### How do you interpret OLS regression results?
### Explain confidence intervals.
### Name the five assumptions of linear regression.
### Estimate the disease probability in one city, given the probability is very low  nationwide. Randomly asked 1000 people in this city, with all negative responses (NO disease). What is the probability of disease in this city?
### What is the difference between linear regression and a t-test? 
### Explain Bayes' Theorem.
### What is bootstrapping?
### How do you inspect missing data, and when are they important?
### What are the underlying assumptions of linear regression, and what are their ### implications for model performance?
### You are asked to reduce delivery delays in a specific geography. How would you apply statistical analysis and machine learning to identify root causes?






## References

https://levelup.gitconnected.com/top-computer-vision-interview-questions-answers-part-3-1e43909131b2 I am not sure if this is too specific. 

https://levelup.gitconnected.com/top-large-language-models-llms-interview-questions-answers-d7b83f94c4e This one gives a good overview for NLP and LLM  I think.


Most important: https://www.teamrora.com/the-2024-technical-interview-guide-for-ai-researchers 




https://www.youtube.com/watch?v=mXki_lLKogM 
https://www.youtube.com/watch?v=NUlLDY16QZU
https://www.amazon.sg/Generative-AI-System-Design-Interview/dp/1736049143
https://www.telusdigital.com/insights/ai-data/article/fine-tuning-llms
https://www.kdnuggets.com/rag-vs-finetuning-which-is-the-best-tool-to-boost-your-llm-application

