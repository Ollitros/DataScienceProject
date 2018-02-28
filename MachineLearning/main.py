import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets.samples_generator import make_blobs
from sklearn.svm import SVC
from mpl_toolkits import mplot3d
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.learning_curve import learning_curve
from sklearn.learning_curve import validation_curve
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix


data = fetch_20newsgroups()
categories = ['talk.religion.misc', 'soc.religion.christian',
              'sci.space', 'comp.graphics']
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(train.data, train.target)
labels = model.predict(test.data)

mat = confusion_matrix(test.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')

def predict_category(s, train=train, model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]



print('The cosmos is something that is not curbed by human and our consciousness: \n', predict_category('The cosmos is something that is not curbed by human and our consciousness'))
print('God is an illusion of peace and quiet. For the people that are afraid of the truth\n', predict_category('God is an illusion of peace and quiet. For the people that are afraid of the truth'))
print('Religion is a doctrine of God that helps to quickly forget reality and live like a blind cat in the lie about the world\n', predict_category('Religion is a doctrine of God that helps to quickly forget reality and live like a blind cat in the lie about the world'))
print('Graphics is the way to tell something to anyone or is used everywhere\n', predict_category('Graphics is the way to tell something to anyone or is used everywhere'))
# plt.show()