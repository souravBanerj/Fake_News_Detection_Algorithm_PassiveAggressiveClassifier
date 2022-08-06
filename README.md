Table of contents<br>
Importing data & exploration
Data cleaning / Prepping
Feature extraction
Model training
Further exploration
Conclusion


# Fake_News_Detection_Algorithm_PassiveAggressiveClassifier
The model I've chosen to use is the Passive-Aggressive (PA) Classifier (see original paper here). In essence, the PA classifier is an algorithm that only updates its weights ("aggressive" action) when it encounters examples for which its predictions are wrong, but otherwise remains unchanged ("passive" action).
Before getting into the actual feature extraction, I want to add some explanations to the method that was used here and why. This may be the most text-heavy section, but I believe it's also crucial to be able to reason the use of your choice of methodology, so please bear with me! But if you so wish, you could also just skip to the "TL;DR" below.

Some terms to know when dealing with... terms
Among a few common ways of extracting numerical features from text are tokenizing, counting occurrence, and tf-idf term weighting; I've chosen tf-idf term weighting here as the feature to extract from these text.

(1) Term frequency
The first portion of this method, "tf", refers to the term frequency, which simply indicates how often terms can be found in documents. Tf's alone are often insufficient as features, however; since there are many commonly-used words such as "is", "are", "the", etc. that do not carry much information about the document, we do not want to weigh these terms as heavily as other more rare but more informative terms. These uninformative terms are actually referred to as stop words, and are often cleaned out during data cleansing/feature extraction steps as they do not hold much value in enhancing the model's ability to predict information.

(2) Inverse document frequency
This is where the "idf", short for inverse document frequency, comes into play. Idf is used to penalize such terms that occur commonly across different contexts without adding interesting information. The exact equation for computing inverse document frequency is:

idf(t)=log1+n1+df(t)+1.
Here, n represents the total number of documents, t represents the term in question, df(t) represents the document frequency of that term; i.e., the number of documents within the set of documents that contain that term. As one can imagine, for common terms such as "is", "are", etc., idf(t) will most likely be 1, since all documents are highly likely to contain them (thus, df(t)=n). On the other hand, the less often a term occurs across different documents, the smaller the denominator will be, making the fraction bigger and in turn, idf(t) bigger.

(3) Tf-idf
Finally, tf-idf is the product of term-frequency and inverse document frequency, mathematically computed as:

tf−idf(t,d)=tf(t,d)∗idf(t).
Where in addition to notations used above, d represents a document. The more commonly the word appears, the greater the value of tf will be, but if this is the case across different documents, it will be penalized with a small idf. On the other hand, a rarely-occurring word might have a smaller value of tf, but be highlighted by bigger idf values for not occurring often in different documents.

TL;DR
Tf-idf term weighting lets you assign importance to tokens that actually carry some information by balancing overall token frequency with its frequency across documents.

Below, I first initialize a TfidfVectorizer object. It takes as input the set of document strings and outputs the normalized tf-idf vectors; then, using fit_transform like any other transformers and predictors in scikit-learn, we can fit the vectorizer to data and tranform them. It has an option to use the max_df to indicate the cut-off document-frequency for stop words, if being used. Here, I will set the cut-off document-frequency to be 0.7, which is the lowest possible value that this parameter can take. The final output of fitting & transforming data will give a sparse matrix with the size of n_samples by n_features, i.e., number of documents by number of unique words.
