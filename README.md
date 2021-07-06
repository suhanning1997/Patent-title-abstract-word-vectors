# Patent-title-abstract-word-vectors
Download Link (Skipgram trained model, word vectors, training information included):
Download Link (Continuous Bag of Words (CBOW) trained model, word vectors, training information included):
In this project, I trained a skip-gram model with negative sampling on the text corpus consisting of all the title + abstraction texts of all patents that exist in the Patentview database (up to 2021). The training procedure (including codes) is outlined in the Word_embedding.py. 

Simplistic intro to the Skipgram model with negative sampling (Mikolov et al. 2013):

Basically, we define a new dataset that consists of positive training examples and negative training examples (0-1 encoding is used). The positve examples are defined as pairs of a target word and its context words. For negative examples, we pair up each context word with multiple randomly selected words from text corpus. We then define a logistic regression with the task of predicting whether the pair is a context-target pair (a positive example) or not. 
In mathematical notation, given any pair of word c and t and $y := \mathbb{I} (\text{c and t form a positive pair})$, we model is :

$$\mathbb{P} (y = 1 | c, t) = \sigma(\bf{\theta}_t^T \bf{e_c})$$

Using appropriate cost function and gradient descent, we train the model and keep the $\bf{e}_c$ as word vectors.

The hyperparameters setting of Li et al (2018). A simple evaluation based on cosine similarity shows that similar words are close in the vector space. (See Word vectors evaluations.ipynb)

Li, Shaobo, et al. "DeepPatent: patent classification with convolutional neural networks and word embedding." Scientometrics 117.2 (2018): 721-744.
Mikolov, Tomas, et al. "Distributed representations of words and phrases and their compositionality." arXiv preprint arXiv:1310.4546 (2013a).\\
Mikolov, Tomas, et al. "Efficient estimation of word representations in vector space." arXiv preprint arXiv:1301.3781 (2013b).\\
