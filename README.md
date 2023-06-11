<div align="center">
      <h1>  <img src="https://github.com/ahammadmejbah/IBM-Project-02-Transform-Photos-to-Sketches-and-Paintings-with-OpenCV/blob/main/Additional%20Files/SN_web_lightmode.svg" width="300px"><br/>IBM Project 06 Sentiment Analysis of Yelp Business Reviews</h1>
     </div>


### Introduction: 
**Sentiment analysis** is the process of computationally identifying and categorizing opinions expressed in a piece of text.  It uses Natural Language Processing (NLP), machine learning, and other data analysis techniques to identify and categorize these opinions.

Sentiment analysis is often used by businesses to gain insight into how their customers feel about their product or service. With that feedback, they can make decisions to improve their customers’ experience.

In this guided project we will use customer reviews from the yelp database. **Yelp** is a local business directory service and review site with social networking features. It allows users to give ratings and review businesses. The review is usually short text consisting of few lines with about hundred words. Often, a review describes various dimensions about a business and the experience of a user with respect to those dimensions.

#### **How Does Sentiment Analysis Work?**
Sentiment analysis algorithms fall into one of the three categories: **rule-based**, where the systems automatically perform sentiment analysis based on the pre-set rules (eg. various NLP techniques like stemming, tokenization, parsing etc.); **automatic**, where the systems rely on machine learning to learn from data, and **hybrid** systems combine both techniques.  In this project we will use machine learning classifier to work with our data.
The basic principle of sentiment analysis is described in the diagram below.

![](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/classification-of-yelp-restaurant-reviews-using-sentiment-analysis/images/Classifier_Algorithm-2.png)

In the **Training Phase**, the feature extractor transfers the text input into feature vectors. Then, the classifier algorithm learns to associate the training text to the corresponding output (eg. label) based on the training data. Pairs of feature vectors and labels are fed into the machine learning algorithm to generate a model.  In the **Prediction Phase**, the feature extractor is used to transform the new document into feature vectors.  These vectors are then fed into the model, which generates predicted labels, in other words, it performs classification.

#### **Feature Extraction from Text**
Machine learning algorithms operate on a numeric feature space, expecting input as a two-dimensional array, where rows are instances and columns are features. In order to perform machine learning on text, we need to transform our text into vector representations. This process is called **feature extraction**, or **vectorization**, and it is an essential first step in language analysis.

Some of the common sentiment encoding approaches are *Bag-of-Words (BOW)*, *Bag-of-Ngrams*, and *Word Embeddings* (also known as word vectors).  To vectorize a corpus (collection of written text) with a bag-of-words approach, we represent every document from the corpus, as a vector, whose length is equal to the vocabulary of the corpus. The computation can be further simplified by sorting token positions of the vector into alphabetical order, as shown in the diagram below. Alternatively, we can keep a dictionary that maps tokens to vector positions. Either way, we arrive at a vector mapping of the corpus that enables us to uniquely represent every document. This article, ["A Gentle Introduction to the Bag-of-Words Model” ](https://machinelearningmastery.com/gentle-introduction-bag-words-model/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsclassificationofyelprestaurantreviewsusingsentimentanalysis29854152-2022-01-01) contains more information about the *Bag-of-Words* approach.

![](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/classification-of-yelp-restaurant-reviews-using-sentiment-analysis/images/CountVectorizer.png)

#### **Tokenizing text with scikit-learn**


To perform feature extraction and vectorization, we first need to tokezine the text.  **Tokenization** is the process of converting text into tokens before transforming it into vectors.  There are many ways to tokenize text, some of them are included in various python libraries. In this guided project, we will use `*CountVectorizer()*` and `TfidfTransformer()` from `scikit-learn` library to tokenize/ vectorize our text.

`CountVectorizer()` can be imported from the `sklearn.feature_extraction model`.  It has its own internal pre-processing, tokenization and normalization methods. It is used to transform a given text into a vector on the basis of the frequency count of each word that occurs in the entire text. So, the value of each cell is nothing but the count of the word in that particular text sample. This link, from [Scikit-Learn Documentation](https://scikitlearn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsclassificationofyelprestaurantreviewsusingsentimentanalysis29854152-2022-01-01), contains more information on `CountVectorizer()`.

`CountVectorizer()` transformer does not take into the account the context of the corpus. Another approach, to consider the relative frequency of tokens in the document against their frequency in the other documents, would be to use the *Term Frequency–Inverse Document Frequency transformer (TF-IDF)*. It can also be imported from the the `sklearn.feature_extraction model`.

`TfidfTransformer()` normalizes the frequency of tokens in a document with respect to the rest of the corpus. This approach accentuates the tokens that are very relevant to a specific document, as shown in the diagram below, since they only appear in that document. `TfidfTransformer()` can be a better choice if more meaning needs to be derived from a particular sentiment. Visit [Scikit-Learn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsclassificationofyelprestaurantreviewsusingsentimentanalysis29854152-2022-01-01#sklearn.feature_extraction.text.TfidfTransformer) to learn more about `TfidfTransformer()`.

![](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/classification-of-yelp-restaurant-reviews-using-sentiment-analysis/images/TF-IDF_Vectorizer-2.png)


``` python
import warnings 
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import  TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score



```

#### Data Cleaning
Next, we want to select only 'stars' and 'text' columns for our analysis. These two columns provide all the necessary information for the purpose of our analysis.

``` python
text = reviews[['stars','text']]
text.head()
```
In the cells below, we are performing some text cleaning. We will do so by defining a function that can remove stopwords and punctuation, convert to lower case, and keep only English reviews. After, we will make a copy of our 'cleaned' data to apply our function to.

``` python
cachedStopWords = stopwords.words("english")
​
def remove_punc_stopword(text):
​
    remove_punc = [word for word in text.lower() if word not in string.punctuation]
    remove_punc = ''.join(remove_punc)
    return [word for word in remove_punc.split() if word not in cachedStopWords]
```

#### Author
Svitlana Kramar
​
Copyright © 2020 IBM Corporation. All rights reserved.
<!-- </> with 💛 by readMD (https://readmd.itsvg.in) -->
    
