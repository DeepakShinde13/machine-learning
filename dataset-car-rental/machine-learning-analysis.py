import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = pd.read_csv("data-for-machine-learning.csv")
print data.head()
nans = data.shape[0] - data.dropna().shape[0]
print ("%d rows have missing values in the train data" % nans)

print data.isnull().sum()

# Make_Model_code has 96 null values
# filling it with some random data

data.MAKE_MODEL_CODE.value_counts(sort=True)
data.MAKE_MODEL_CODE.fillna('FORD', inplace=True)
print data.isnull().sum()

# load sklearn and encode all object type variables
for i in data.columns:
    if data[i].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(data[i].values))
        data[i] = lbl.transform(list(data[i].values))

##now everything is encoded


# ACOR_TOTAL values with 0 are should be
# replaced by some random data with less than $500

y = data.ACOR_TOTAL
x = data.drop('ACOR_TOTAL', axis=1)  # remaining data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print "\nX_train:\n"
print(x_train.head())
print x_train.shape
print "\nX_test:\n"
print(x_test.head())
print x_test.shape
print "y_test"
print (y_test.head())
print "y_train"
print (y_train.head())

# Define a function to create the scatterplot. This makes it easy to
# reuse code within and across notebooks
def scatterplot(x_data, y_data, x_label, y_label, title):

    # Create the plot object
    _, ax = plt.subplots()

    # Plot the data, set the size (s), color and transparency (alpha)
    # of the points
    ax.scatter(x_data, y_data, s = 30, color = '#539caf', alpha = 1)

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.show()

# Call the function to create plot
scatterplot(x_data = data['MAKE_MODEL_CODE'], y_data = data['ACOR_TOTAL'], x_label = 'MODEL_YEAR', y_label = 'Amount', title = 'Graph ')

# visualising the data
#plt.plot(y, x)
#plt.show()

# from sklearn.ensemble import RandomForestClassifier
# #train the RF classifier
# clf = RandomForestClassifier(n_estimators = 500, max_depth = 6)
# clf.fit(x_train,y_train)
#
# RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#              max_depth=6, max_features='auto', max_leaf_nodes=None,
#              min_impurity_split=1e-07, min_samples_leaf=1,
#              min_samples_split=2, min_weight_fraction_leaf=0.0,
#              n_estimators=500, n_jobs=1, oob_score=False, random_state=None,
#              verbose=0, warm_start=False)

from sklearn.svm import SVC
clf = SVC()
clf.fit(x_train, y_train)
SVC(C=1.0, cache_size=200, coef0=0.0, degree=3, gamma='auto', kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

prediction = clf.predict(x_test)

acc = accuracy_score(np.array(y_test), prediction)
print "HI"
print acc
