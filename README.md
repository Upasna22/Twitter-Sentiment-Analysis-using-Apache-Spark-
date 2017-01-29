# Twitter-Sentiment-Analysis-using-Apache-Spark-
Accessed the Twitter API for live streaming tweets. Performed Feature Extraction and transformation from the JSON format of tweets using machine learning package of python pyspark.mllib. Experimented with three classifiers -Naïve Bayes, Logistic Regression and Decision Tree Learning and performed k-fold cross validation to determine the best.



# Twitter Sentiment Classification using Apache Spark

## TWEET PROCESSING:

### Describe the tweet processing steps.

In order to process the given tweets in train.csv , essentially we first need to clean the dataset (train.csv).

#### Cleaning :

Firstly , the csv file is read with a csv file reader. The polarity of the tweets and the tweet text itself are extracted from the csv and stored in two seperate lists. We extract each word from the tweet text in order to do further cleaning. This is achieved by an easy spilt with a space i.e, split(" "). The words obtained by splitting is stored in another word list for easy access.

According to the suggestions of the assignment , the first thing we did is to lowercase all words using the string.lower() function. The replacing of  @someuser with AT_USER is done next by searching for an @ and replacing accordingly. Digits are removed for better cleaning and accuracy. To remove stopwords from our text we defined a stopwords list. If any word of the word-list appears in the stopwords-list , we defined a function "remove_values_from_list( the_list , val )" which returned "the_list" as long as the value is not equal to "val"( stopwords-list values).

Next , we remove punctuations of the words in word-list by stripping from the words string.punctuation. The polarity or label and cleaned text-list is shuffled to get labels and text in a random order( this is done because majority of the data in train.csv has the first 1000 so lines with 0 label and then lines with a 1 label. So to get text with 0 and 1 labels uniformly shuffled we use the random.shuffle function).

#### Convert the cleaned data to RDD:

According to the documentation there are two ways to create RDDs: parallelizing an existing collection in your driver program, or referencing a dataset in an external storage system

We used the parallelizing method to convert our cleaned list to an RDD.
```
sc.parallelize(rddname)
```
#### Convert to LabeledPoint :

In order to proceed with training , the Naive Bayes , Logistic Regression and Decision tree models take LabeledPoint data as input to preform classification. Therefore we define a function  "def Convert_to_LabeledPoint(labels,features) ".

For Training our data with the Naive Bayes , Logistic Regression and Decision tree models , we first define functions for computing the TF , IDF and TFIDF.

#### Computing Term Frequency :

According to the documentation : "TF and IDF are implemented in HashingTF and IDF. HashingTF takes an RDD of list as the input. Each record could be an iterable of strings or other types."

Therefore we input our RDD into the HashingTF function which is defined as follows:
```
hashingTF = HashingTF()
tf = hashingTF.transform(t_rdd)
```
#### Computing Inverse Document Frequency:

Similarly , for the IDF applying IDF needs two passes: first to compute the IDF vector and second to scale the term frequencies by IDF and is calculated as follows as specified in the documentation :
```
def CompIDF(tf):
    tf.cache()
    idf = IDF().fit(tf_train)
    return idf
```
#### Computing TF-IDF: 

We define a function CompTFIDF to compute the TFIDF by following the below code:
```
tf.cache()
idf = IDF(minDocFreq=2).fit(tf)
tfidf = idf.transform(tf)
```
NOTE: We use TF-IDF score in all our models for training and testing.

#### TRAINING :

We have used 3 models for training as explained below :

**Naive Bayes :**

Naive Bayes is a simple multiclass classification algorithm with the assumption of independence between every pair of features. Naive Bayes can be trained very efficiently. Within a single pass to the training data, it computes the conditional probability distribution of each feature given label, and then it applies Bayes’ theorem to compute the conditional probability distribution of label given an observation and use it for prediction.

**Logistic Regression :**

Logistic Regression is widely used to predict a binary response.

**Decision Trees:**

Decision trees are widely used since they are easy to interpret, handle categorical features, extend to the multiclass classification setting, do not require feature scaling, and are able to capture non-linearities and feature interactions.

**Training :**

**To Train :**

* First compute the TF using CompTF()
* Next compute the IDF using CompIDF()
* Compute TFIDF using CompTFIDF()
According to the documentation the classifiaction algorithms take an RDD of LabeledPoint and an optionally smoothing parameter lambda as input, and outputs a model which can be used for evaluation and prediction.

Therefore ,we use the cleaned and shuffled dataset and convert it to an RDD by using sc.parallelize. The RDD is then converted to a LabeledPoint format by using the Convert_to_LabeledPoint() function defined before. Now we have the right format to train our data. spark.mllib provides convenient APIs to train with .

#### Train a naive Bayes model.
```
model = NaiveBayes.train(training)
```
#### Train with a Logistic Regression model.
```
model = LogisticRegressionWithLBFGS.train(parsedData)
```
#### Train with a Decision Tree model.
```
model = DecisionTree.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={}, impurity='gini', maxDepth=5, maxBins=32)
```
 
#### FEATURE SPACE :   

### Describe the feature space. Did you decide to use unigrams, bigrams, or both? What is the size of your feature space?

Essentially, the feature space consists of the set of features that are used for training and classifying the data.
The various methods used are HashingTF() and IDF(). The HashingTF function in mllib has an attribute called as numFeatures, where we can vary the number of features that our model is going to use.

For **Naïve Bayes**,

The feature space consists of 25000 features and gives us a test accuracy of 74.37%.

For **Logistic Regression**,

The feature space consists of 25000 features and gives us a test accuracy of 77.99%.

For **Decision Tree Learning**,

The feature space consists of 1000 features and gives us a test accuracy of 60.72%.

We have used unigrams for our feature space.
 

#### EXTRA WORK :

### Describe any extra work (i.e. parameter tuning) you did on three classifiers: NB, LOG and Decision Tree(DT). How did it help?

As far as parameter tuning is concerned, we have experimented with various numFeature spaces and obtained different results for all the classifiers.

For **Naïve Bayes**,
When all the features in the feature space is being used, the accuracy or performance of the model is not that good, the accuracy obtain is around 72% but when we reduced the feature space to 50000 and obtained a test accuracy of 73%, at this point we assumed that for a lower sublet of features, the accuracy increases and therefore we used a feature space of 25000 and obtained a test accuracy of 74.37%.

For **Logistic Regression**,
We used a similar approach for Logistic Regression, where the feature space was varied and we settled on 25000 and obtained a test accuracy of 77.99%.

For **Decision Tree Learning**,
In the case of Decision Trees, when we used  the entire feature space we observed that it was taking a large amount of time to process. This means that the decision tree was growing exponentially due the large feature space, we also tried with several other numbers like 50000, 25000, 15000 and we faced the same problem.

Towards the end we input a feature space of 1000 and obtained a test accuracy of 60.72%.

#### Findings :

Here , we understand that Logistic Regression has the highest Accuracy as well as Precision and Recall . By this we can infer that Logistic regression is more robust and efficeint in classifying binary labels.

Whereas , Decision Trees seem to give us a low Accuracy. Basically this is because we have given a limited feature space to the classifier.

According to the graphs, it appears that the Naïve Bayes Classifier overfits the most. Essentially, we have come to this conclusion because the Training accuracy of Naïve Bayes is 81.29125 % and the Test accuracy of Naïve Bayes is 74.3732590529248 % .The difference is large which indicates that the classifier has classified the training data better than the test data, and therefore we can say that Naïve Bayes over-fits.

### PRECISON , RECALL , F1-SCORE :

#### Describe the following terms in the context of the assignment: precision, recall, f1-score, confusion matrix (true positive, true negative, false positive, false negative).

**Precision :**
In our assignment we can describe the precision for a class as the number of true positives (i.e. the number of items correctly labeled as belonging to the positive class) divided by the total number of elements labeled as belonging to the positive class (i.e. the sum of true positives and false positives, which are items incorrectly labeled as belonging to the class).

**Recall :**
Recall can be described as the number of true positives divided by the total number of elements that belong to the positive class (i.e. the sum of true positives and false negatives, which are items which were not labeled as belonging to the positive class but should have been).

**F1 Score :**
The F1 score is a measure of our test's accuracy.It will consider both the precision and the recall of the test to compute the score.It can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst at 0.

**Confusion Matrix:**
The Confusion Matrix allowed us to visualize the performance of an algorithm. Each column of the matrix represents the instances in a predicted class while each row represents the instances in an actual class (or vice-versa).

### ROC CURVE :

**Area under ROC_LG** = 0.7788539144471348

**Area under ROC_NB** = 0.7435276587818961

Here, to get the ROC curve of a Naïve bayes it is impossible just by using mllib in pyspark, there are other ways to do it using other libraries, we have tried exploring several options which involved getting into the pyspark source code and making changes which would not have been practical because we cannot reproduce our output in another system, since we have limited resources, we have considered the labels and predictions of the training accuracy and testing accuracy and computed the False Positive Rates and True Positive Rates, and plotted the ROC curve, this may not be a right way to do it but the testing the classifier on different datasets can be considered to be a threshold.
The area under ROC curve is 0.7435276587818961 (Which is correct)

#### Learning from ROC Curves :

An ROC curve shows the tradeoff between sensitivity and specificity (any increase in sensitivity will be accompanied by a decrease in specificity). We can say that the test is accurate if the curve lies close to the left-border and the top-border of the graph. When we compare the ROC curves of Logistic Regression and Naive Bayes ,it is noticeable that the curve of  LR nbsp; closely follows the left and top border and thereby is more accurate.

Also , the 45 degree diagonal of the ROC space denotes accuracy. The closer the curve to the diagonal the lesser the accuracy. The Logistic Regression curve is less closer to the 45 degree diagonal than the Naive Bayes curve. Hence it is more nbsp; accurate.

### BEST CLASSIFIER :

#### Which classifier performs the best? Why?

Most accurate based on highest 10-fold CV accuracy :- LG is the most accurate Classifier.
Most accurate based on average 10-fold CV accuracy :- LG is the most accurate Classifier

According to our findings we understand that Logistic Regression gives us the highest accuracy.

Some of the reasons are, because Logistic Regression gave us convenient probability scores for observations, we used L2 regularization, the feature space is used wisely since it takes each feature in a stepwise manner and computes probabilities.

### About this project

This project was done as a part of CS 491: Introduction to Data Science at UIC
