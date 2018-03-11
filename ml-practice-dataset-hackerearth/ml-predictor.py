#make prediction and check model's accuracy
import pickle
import numpy as np
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#temp = "20 3 109813 1 7 4 12 2 4 1 0 0 40 38"#
temp = sys.argv[1]
#temp = "28, Local-gov,336951, Assoc-acdm,12, Married-civ-spouse, Protective-serv, Husband, White, Male,0,0,40, United-States"
print temp, "USER_INPUT"
test_data = np.fromstring(temp, dtype=int, sep=',')
test_data = np.reshape(test_data, (-1, len(test_data)))
print test_data
filename  = '/home/deepak/Desktop/finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# result = loaded_model.score(X_test, Y_test)
# print(result)


#20,3,109813,1,7,4,12,2,4,1,0,0,40,38 data to be passed
prediction = loaded_model.predict(test_data)
print prediction
with open("/home/deepak/Desktop/result.txt", 'wb') as target:  # specify path or else it will be created where you run your java code
    target.write(prediction)