from numpy.core.fromnumeric import ravel
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
import statistics as st
import random as rd
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split as tts 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score as AS
from sklearn.metrics import confusion_matrix as CXM

data_file = pd.read_csv("c117/BankNote_Authentication.csv")

variance =  data_file["variance"]
data_class =  data_file["class"]

variance_train,variance_test,data_class_train,data_class_test =  tts(variance,data_class,test_size=0.25,random_state=0)

# to train the data using reshape()
X = np.reshape(variance_train.ravel(),(len(variance_train),1))
Y = np.reshape(data_class_train.ravel(),(len(data_class_train),1))

data_classifier = LogisticRegression(random_state=0)
data_classifier.fit(X,Y)

X_test = np.reshape(variance_test.ravel(),(len(variance_test),1))
Y_test = np.reshape(data_class_test.ravel(),(len(data_class_test),1))

y_predict = data_classifier.predict(X_test)
predict_values = []

for i in y_predict:
    if i == 0:
        predict_values.append("Authorized")
    else:
        predict_values.append("Forged")
        
actual_values = []

for x in Y_test.ravel():
    if x == 0:
        actual_values.append("Authorized")
    else:
        actual_values.append("Forged")
        
        
# to plot graph data

labels = ["no","yes"]

cm_result = CXM(actual_values,predict_values )

graph_heat_map  = plt.subplot()
sns.heatmap(cm_result, annot=True, ax = graph_heat_map)
graph_heat_map.set_xlabel("predict")
graph_heat_map.set_ylabel("actual")
graph_heat_map.set_title("confusion matrix")
graph_heat_map.xaxis.set_ticklabels(labels); graph_heat_map.yaxis.set_ticklabels(labels)
# plt.show()

true_positve = 1.2e+02
false_positve = 27
true_negative = 1.7e+02
false_negative = 29

result = (true_positve+true_negative)/(true_positve+true_negative+false_negative+false_positve)
print(result)
