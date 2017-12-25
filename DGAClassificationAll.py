__author__ = 'andrewa'

import os
import math

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix


from collections import Counter




#dataframe_dict = pd.DataFrame.from_dict(dataframe_dictiontary,orient='index').transpose()

"""
alexa
conficker
cryptolocker
zeus
pushdo
rovnix
tinba
matsnu
ramdo
"""




def entropy(s):
    """

    :param s:
    :return:
    """
    p, lns = Counter(s), float(len(s))
    return -sum(count / lns * math.log(count / lns, 2) for count in p.values())



def parse_files():
    """

    :return:
    """
    print("Creating dataframe")

    dataframe_dict=dict()


    for file in os.listdir("dga_wordlists/"):
        if file.endswith(".txt"):
            print(os.path.join("dga_wordlists", file))

    #dataframe_dict= pd.DataFrame()

    for file in os.listdir("dga_wordlists/"):

        if file == 'legit_domains.txt':
            print("Parsing alexa")
            v = pd.read_csv(os.path.join("dga_wordlists", file), names=['uri'], header=None, encoding='utf-8')

            #v['tld'] = v.applymap(lambda x: x.split('.')[1].strip().lower())
            v['domain'] = v.applymap(lambda x: x.split('.')[0].strip().lower())
            v['class'] = 'legit'
            del v['uri']

            try:
                dataframe_dict[file] = v
                print(type(v))
            except Exception as e:
                print("exception: {}".format(e))

        else:
            print("Parsing malware {}".format(file))
            v = pd.read_csv('dga_wordlists/' + file, names=['uri'], header=None, encoding='utf-8')
            print(dir(v))
            v['domain'] = v.applymap(lambda x: x.split('.')[0].strip().lower())
            v['class'] = 'dga'
            del v['uri']

            dataframe_dict[file] = v
            print(type(v))

    print('# done parsing')
    return dataframe_dict

dataframe_dict = parse_files()


#serie_to_concat = [i for i in dataframe_dict.items()  ]
#print(len(serie_to_concat), "is the size of the concatenation")
print(type([dataframe_dict.items()]))
a= list(dataframe_dict.items())[0]

print(a)
input()

all_domains = pd.concat(dataframe_dict, ignore_index=True)
print(all_domains)


print("Calculating length")
all_domains['length'] = [len(x) for x in all_domains['domain']]

print("Calculating entropy")
#all_domains['entropy'] = [entropy(x) for x in all_domains['domain']]

# all_domains.boxplot('length','class')
# pylab.ylabel('Domain Length')
# all_domains.boxplot('entropy','class')
# pylab.ylabel('Domain Entropy')

# cond = all_domains['class'] != 'legit'
# dga = all_domains[cond]
# alexa = all_domains[~cond]
# plt.scatter(alexa['length'], alexa['entropy'], s=140, c='r', label='Legit', alpha=.4)
# plt.scatter(dga['length'], dga['entropy'], s=140, c='#aaaaff', label='DGA', alpha=.4)
# plt.legend()
# pylab.xlabel('Domain Length')
# pylab.ylabel('Domain Entropy')
# plt.show()

print("vectorizing, strpping, antaning")
alexa_vc = CountVectorizer(analyzer='char', ngram_range=(3, 5), min_df=1e-4, max_df=1.0)
counts_matrix = alexa_vc.fit_transform(dataframe_dict['legit_domains.txt']['domain'])
alexa_counts = np.log10(counts_matrix.sum(axis=0).getA1())

dict_vc = CountVectorizer(analyzer='char', ngram_range=(3, 5), min_df=1e-5, max_df=1.0)
word_dataframe = pd.read_csv('help/words.txt', names=['word'], header=None, dtype={'word': np.str}, encoding='utf-8')
word_dataframe = word_dataframe[word_dataframe['word'].map(lambda x: str(x).isalpha())]
word_dataframe = word_dataframe.applymap(lambda x: str(x).strip().lower())
word_dataframe = word_dataframe.dropna()
word_dataframe = word_dataframe.drop_duplicates()
counts_matrix = dict_vc.fit_transform(word_dataframe['word'])
dict_counts = np.log10(counts_matrix.sum(axis=0).getA1())

all_domains['alexa_grams'] = alexa_counts * alexa_vc.transform(all_domains['domain']).T
all_domains['word_grams'] = dict_counts * dict_vc.transform(all_domains['domain']).T
all_domains['diff'] = all_domains['alexa_grams'] - all_domains['word_grams']

print('Done data')



def plot_cm(cm, labels):
    percent = (cm*100.0)/np.array(np.matrix(cm.sum(axis=1)).T)  # Derp, I'm sure there's a better way
    print('Confusion Matrix Stats')
    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            print("%s/%s: %.2f%% (%d/%d)".format((label_i, label_j, (percent[i][j]), cm[i][j], cm[i].sum())))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(b=False)
    cax = ax.matshow(percent, cmap='coolwarm')
    pylab.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    pylab.xlabel('Predicted')
    pylab.ylabel('True')

    pylab.show()




def classify(X_train, X_test, y_train, y_test):
    """

    :return:
    """

    clf1 = LogisticRegression(random_state=1)
    clf1.fit(X_train, y_train)

    clf2 = RandomForestClassifier(bootstrap=True, max_depth=None, class_weight="auto", min_samples_leaf=1,
                                  min_samples_split=1, n_estimators=1500, n_jobs=40, oob_score=False,
                                  random_state=1, verbose=1)
    clf2.fit(X_train, y_train)

    clf3 = GaussianNB()
    clf3.fit(X_train, y_train)

    clf4 = ExtraTreesClassifier()
    clf4.fit(X_train, y_train)

    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3), ('etr', clf4)], voting='soft')

    for clf, label in zip([clf1, clf2, clf3, clf4, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes',
                                                           'Extra Tree', 'Ensemble']):
        scores = cross_val_score(clf, X_test, y_test, cv=5, scoring='accuracy')
        print("Accuracy: %0.6f (+/- %0.2f) [%s]".format((scores.mean(), scores.std(), label)))

    return clf2


if __name__ == "__main__":
    print("Beginning")


    X = all_domains.as_matrix(['length', 'entropy', 'alexa_grams', 'word_grams', 'diff'])
    y = np.array(all_domains['class'].tolist())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.31337)

    classify(X_train, X_test, y_train, y_test)

    clf2 = RandomForestClassifier(bootstrap=True, max_depth=None, class_weight="auto", min_samples_leaf=1,
                                  min_samples_split=1, n_estimators=1500, n_jobs=40, oob_score=False,
                                  random_state=1, verbose=1)
    clf2.fit(X_train, y_train)

    y_pred = clf2.predict(X_test)
    labels = ['legit', 'dga']
    cm = confusion_matrix(y_test, y_pred, labels)

    plot_cm(cm, labels)













# cond = all_domains['class'] != 'legit'
# dga = all_domains[cond]
# legit = all_domains[~cond]
# plt.scatter(legit['length'], legit['word_grams'],  s=140, c='r', label='legit', alpha=.1)
# plt.scatter(dga['length'], dga['word_grams'], s=140, c='#aaaaff', label='DGA', alpha=.1)
# plt.legend()
# pylab.xlabel('Domain Length')
# pylab.ylabel('Dictionary NGram Matches')
# plt.show()

# cond = all_domains['class'] != 'legit'
# dga = all_domains[cond]
# legit = all_domains[~cond]
# plt.scatter(legit['length'], legit['diff'], s=140, c='r', label='Legit', alpha=.4)
# plt.scatter(dga['length'], dga['diff'], s=140, c='#aaaaff', label='DGA', alpha=.4)
# plt.legend()
# pylab.xlabel('Domain Length')
# pylab.ylabel('Diff')
# plt.show()

# weird_cond = (all_domains['class']=='legit') & (all_domains['word_grams']<3) & (all_domains['alexa_grams']<2)
# weird = all_domains[weird_cond]
# all_domains.loc[weird_cond, 'class'] = 'weird'
# not_weird = all_domains[all_domains['class'] != 'weird']
# X = not_weird.as_matrix(['length', 'entropy', 'alexa_grams', 'word_grams'])
# y = np.array(not_weird['class'].tolist())