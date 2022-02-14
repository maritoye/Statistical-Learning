# Statistical Learning Project
<b>Project in ACIT4510 - Statistical Learning, OsloMet</b> </br>
<b>Marit Øye Gjersdal, 2021</b>

Political opinions; an aspect of our lives that are so personal, yet arguably one of the most important features about us for the society to run smoothly in a democratic country. Every other year in Norway it is time for a new election. All the political parties line up in the streets months ahead in order to convince the voters that they should get their precious vote. Countless hours are spent by experts discussing on TV why they believe the voters will go for one party or another, and what will be the core cases that will turn the voters from one party to another. The news are blowing up with new polls every week, all chasing the answer for the ultimate question; who will we vote for? 

This project intends to explore whether it will be possible to predict a persons political standpoint based on their stated opinions on the living conditions and services of the area they live in as well as stated info about their background, income, work, education, and family/living situation.

I will use classification methods in order to predict who a person voted for in the Norwegian 2015 municipal election. The methods I will be using range from simpler methods such as tree classification, all the way to the more advanced neural network models. 


## Data
The dataset used for this project is from a survey called [Innbyggerundersøkelsen 2019](https://search.nsd.no/study/41257c0f-8e13-4ff4-80d1-c50df7952c52), or the citizen survey, which is a survey conducted by the Norwegian Agency For Public Management And EGovernment (called DIFI in Norwegian). [The survey](https://www.nsd.no/data/individ/publikasjoner/NSD2883/sporreskjema_innbyggerundersokelsen_om_a_bo_i_kommunen.pdf) aims to map out how satisfied the residents are with many different aspect of their lives, their municipalities, the country as a whole. It covers areas such as healthcare, infrastructure, education and culture, as well as collecting data about the participants health, family and living situation. The creators of the dataset have also written a thorough [report presenting some finding in the data](https://www.nsd.no/data/individ/publikasjoner/NSD2883/innbyggerundersokelsen_2019_-_rapport_innbyggerdel.pdf). The report presents many interesting findings, but does not touch upon the political aspect of the participants responses, such as this project aim to do. 

The dataset documentation is a 150 page document explaining all the variables and which question from the survey they each corresponds to. According to this document the survey had 7134 People across 5 age groups over 18 responding to the survey, and the data was to be of 7134 rows, where the 70 questions were represented in 266 features. The documentation does not mention empty cells, but they have specified numeric values representing "Nan" values for many features. 

The variables are all represented either as a binary value or at a scale of discrete values, most of them from 1 to 8. For most of the non-binary variables the number indicates a low to high value, as the participants were asked to rate their opinions as for example discontent to very content, where in the dataset values 1 represents discontent and 7 represents content. Many of these variables uses 8 to represent the “I don’t know” option in the survey. The dataset also contains a handful categorical variables, such as which political party they voted for in the 2015 municipal election, which is the question we are mainly concerned with in this project as this is the value we wish to predict. The different answer options for these categorical questions are represented as a number, and they have provided a corresponding list of which number corresponds to which category in the documentation document.


### Challenges with the data
As I have already touched upon, the documentation for the dataset happened to be quite misleading. Not only was it difficult enough having to navigate a 150 page document in search for the meaning behind the variable names which are on formats such as "Q65a\_5", a binary value representing one of the options to the question "Do you have any physical or mental disability that limits you in your daily life, and that has lasted or will last for 6 months or more? -Yes, other" (my own translation from Norwegian). There are not provided any table mapping the variable names to the questions. Once finding the correct variable and the list of possible values in the documentation some of this information is in many cases incorrect too. For several variables one specific value was listed to represent "Nan" but when looking at the data I found several instances of other values such as 9999 or just empty cells.

The first issue I encountered with the data was the surprising amount of rows, 50851 to be exact. After inspecting the data further I found that the dataset included not only the data from the 2019 survey as the documentation said, but also the data from the four previous citizen surveys. I later attempted using all five years of data, as this would have been good for training the models, however each years data had slightly different issues and questions missing.

When starting to thoroughly clean all the data I encountered another surprising problem. As described before this project will try to predict which political parties the participants voted for, which is represented in variable "Q16" for the municipal election in 2015 and "Q40a" for the parliamentary elections in 2017. Not for a single one of the 7134 participants had answered both of these questions. The documentation does mention that the survey was in two parts, but completely failed to mention that the two parts were answered by two separate groups of people. (I did also check for Id, just in case each participant was represented in two rows, but all the participant ids were unique.) I fount that there were 53 variables specific only to the municipal part of the survey, and 65 specific to the parliamentary part of the survey. 


### Cleaning
This following section summarises initial cleaning of the data, which I've done in the motebook data_cleaning.ipynb.

I first had to remove the data for all years except 2019, as the rest was never supposed to be there. I then removed the year variable ("Aar"). The dataset contained 95 boolean values representing 3 questions for the survey. As these three questions were only technical questions regarding the survey I removed all these variables, as well as eight other survey technical variables. 

For 20 boolean variables representing three question many values were missing. As it was obvious from the nature of the questions and the placement of the missing values that the missing values were most likely supposed to be zeros, I filled them all with 0. 

Since this project is concerned about predicting the “Q16” (party voted for in the last municipal election), so I removed all rows for participants that had not answered this question, hence removing the participants of the parliamentary part of the survey. Then I removed all the columns that belonged only to the parliamentary part, as they were all empty for the rows that were left. I was now left with 3359 rows and 96 features.


### Imputation of missing values
All the rows in the dataset contained some missing values. Since many models do not accept missing values, and I clearly could not remove rows containing missing values as this would result in en empty set, some other method was needed. 

I therefore chose to use multivariate imputation, as this takes all columns into consideration when calculating the missing values rather than unvariate imputation which only uses the median (or mean or knn) of that variable. The multivariate imputation will calculate the new value as a float. For my dataset all values were integers, but this is fine and I will not round them off, because when training the models I treat them all as continuous numerical.

I made four categories of the types of variables I wanted to do the imputation on: numerical values where 9999 represented “I don’t know”, numerical values where 8 represented “I don’t know”, values of 1 meaning yes or 2 meaning no where 3 represented “I don’t know”, and three categorical values with numerical meaning (that I have chosen to treat as numerical in the models) where number 9, 5 and 9999 respectively represented “I don’t know” in the survey. For all these variable, I first replaces the respective “I don’t know” values with Nan, and then I made a multivariate imputer and transformed all the Nan values. 

## Methods
### Support Vector Machine
For the first model I used [scikit learns Support Vector Classification](https://scikit-learn.org/stable/modules/svm.html). The first version I made of the SVM gave an accuracy of 0.34 when doing 5-fold cross validation, with a standard deviation of 0.01. When testing on the test set it has an accuracy of 0.3089. As seen in Figure \ref{fig:svm}, the model only predicted values 1 and 3 for the entire test set (1 means "Arbeiderpartiet", and 3 means "Høyre"). The reason is that these two values are overrepresented in the dataset compared to the rest. 

I then made a second support vector machine, but this time balancing the data in hopes that the model would be able to predict other categories than 1 and 3 as well. As seen in the figure below, this new balance SVM predicted a much wider range of values/ parties when testing on the test set. However the accuracy was far lower at only 0.19 when doing 5-fold cross validation, with a standard deviation of 0.01, and  0.2278 on the test set.

![Correlation matrices of testing Support Vector Machine trained without and with balancing the target data: SVM (not balanced)](/img/cm_svm.jpg)

![Correlation matrices of testing Support Vector Machine trained without and with balancing the target data: SVM balanced](/img/cm_svm_balanced.jpg)


### Classification tree
Another simple classification model is the [classification tree](https://scikit-learn.org/stable/modules/tree.html), that can learn simple rules inferred from the features to predict. An aspect of the classification tree that intrigued me is the explainability. The fact that we can see clearly why the model predicted the value that was predicted can be very valuable for analysing complex problems such as political orientation. The classification tree gave an accuracy of 0.21 when doing 5-fold cross validation, with a standard deviation of 0.01. When testing on the test set it has an accuracy of 0.2406. A visualization of the calculated feature importances can be found in appendix \ref{imp}.

### Random forest classifier
Since my classification tree did not give very satisfactory results, I wanted to try an improvement: the [random forest classifier](https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees). A large benefit of this is that as with the classification tree, it can calculate feature importance. The random forest classifier significantly improved the accuracy compared to the classification tree. As opposed to the classification tree with its accuracy of 0.21 when doing 5-fold cross validation, the random forest classifies gave an accuracy of 0.34, with a standard deviation of 0.01. When testing on the test set it has an accuracy of 0.3656.

### Multi-layer perceptron
In an attempt to reach a higher accuracy than what my models had achieved this far, I wanted to explore more advanced models, specifically neural networks. The first ANN model I created is an implementation of [scikit learns multi-layer perceptron classifier](https://scikit-learn.org/stable/modules/neural_networks_supervised.html). 

### Self-defined neural network
Since my implementation of the MLPClassifier gave quite unsatisfactory accuracy, in fact lower than two of the previous model I created, I wanted to implement a deep neural network to see if this would be able to predict more accurately. With the help of [this tutorial](https://medium.com/luca-chuangs-bapm-notes/build-a-neural-network-in-python-multi-class-classification-e940f74bd899) I made a neural network with three hidden dense layers. I trained it with the condition to stop when the validation accuracy had not improved for ten epochs.

## Results
The two models that gave the highest accuracies using 5-fold cross validation was the initial implementation of the support vector machine and the random forest classifier, both with an accuracy of 0.34. When testing on the test set though, the random forest classifier gave an accuracy of 0.3656, while the SVM had an accuracy of 0.3089. This does however not necessarily mean that the random forest classifier is more accurate then the SVM, as the test data set was quite small. 

The figure below shows a box plot of the accuracies for the 5-fold cross validation of each of the models i implemented. The table gives the average accuracies of the 5-fold cross validation as well ass the accuracy when testing on the test set for each model. 

![Boxplot](/img/box.png)

| Model                  | n-fold cross validation | test set |
|------------------------|-------------------------|----------|
| Support Vector Machine | 0.34                    | 0.3089   |
| SVM balanced           | 0.19                    | 0.2278   |
| Classification Tree    | 0.21                    | 0.2406   |
| Random Forest          | 0.34                    | 0.3656   |
| Multi-Layer Perceptron | 0.26                    | 0.25     |
| Neural network         | 0.30                    | 0.325    |

Below are the confusion matrices of each of the models when predicting on the test set. From the plots, we can see that the two models with highest accuracy predicted nearly everyone to have voted for 1 and 3, meaning "Arbeiderpartiet" and "Høyre", which are in fact the two largest parties in Norway. The balances version of the SVM was the only model to predict a large range of classes for this test, however it did this at the expense of the accuracy, as this was also the model with lowest accuracy between them all. 

![Confusion Matrices](/img/cm.png)

## Discussion
When beginning this project, I already suspected that trying to accurately predict a persons political view could be a challenging task. I also encountered many unexpected challenges in this project. The main being that the dataset was not as described. It contained only half rows as it says since it was divided into two surveys, and it had tons of missing values. This led to having to spend a enormous amount of time reading each and every question in the questionnare to understand each data point, so I could choose what to do about the missing values or the Nan/"I dont know" values. 

It was also hard to decide which of the (seemingly) categorical values to one-hot-encode. Some were obvious (such as "Fylke"), but others were not (Age categories and income categories) as they were categorical but still held a numerical meaning in relation to one another. 

If I were to have guessed randomly between the ten parties I would end up averaging at about 10 percent accuracy. Had I only guessed the largest political party I would get an average accuracy of somewhere between 21-33 percent, depending on the current votes of the largest party (and of course given that I guessed for a group that was representative for the Norwegian population). Several of the models implemented in the project were able to achieve higher scores than this, which indicates relations between some of the variables and which party a person voted for. 

I approaches this as a classification problem where I saw all the political parties as ten separate categories. When evaluating the models I also did this, only looking at whether the prediction was correct or not correct. 

However these categories are in fact not all as different from one another. Some of the political parties are quite similar, and therefore tend to attract the same type of voters. It is not always apparent for a voter which party they want to vote for, however it is more likely that the alternatives are two (or more) parties that are quite similar. An example from these Norwegian parties would be that a person who ended up voting for "Sosialistisk Venstreparti" might have also considered voting for "Rødt", but would never have even imagined voting for "Fremskrittspartiet". This is not something I have taken into consideration in my approach. 

If we were able to somehow represent how people voted similarly (meaning closer on the political scale) without voting for the same party, then train the models given this numerical meaning between the data, it might improve the models accuracies. 

I therefore suggest the following as a alternative approach to the projects problem: Looking at the problem as a political scale from 1-10, from "Rødt" = 1 to "Fremskrittspartiet" = 10. I would need to sort the parties on the scale according to the normal scale from "red" to "blue" parties, and then make a new column in the dataset where I translated all the old values into the new range. Even though it would be discussable which order to give the parties, the parties would nevertheless get some numerical meaning, and we could then train models to predict a continuous value from 1-10. This would open a whole new range of possible models.

