from msilib.schema import Class
import numpy as np
import pandas as pd
import time
start_time = time.time()
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import category_encoders as ce
from sklearn import preprocessing, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns

class ClassificationObject():
    def __init__(self, score, exec_time):
        self.score = score
        self.exec_time = exec_time

    def get_score(self):
        return self.score

    def get_exec_time(self):
        return self.exec_time


def RDFClassifier(df, nTrees):
    rdf_start_time = time.time()
    X = df.drop(['classe'], axis=1)
    Y = df['classe']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=None)

    rdf = RandomForestClassifier(max_depth=7, n_estimators=nTrees)

    rdf.fit(X_train, Y_train)
    predicted = rdf.predict(X_test)

    rdf_obj = ClassificationObject(accuracy_score(Y_test, predicted)*100,
                                                 time.time() - rdf_start_time)

    return rdf_obj


def KNNClassifier(df):
    knn_start_time = time.time()
    
    X = df.drop(['classe'], axis=1)
    Y = df['classe']

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.33, random_state=None)  # 1/3 da base de dados reservada para testes

    # a fim de achar o numero ótimo de Kneighbors, temos a raiz quadrada do numero total de carros(linhas) do df
    k = np.sqrt(df.size/7)
    k = int(k)

    knn = KNeighborsClassifier(n_neighbors=k).fit(X_train, Y_train)

    predicted = knn.predict(X_test)

    knn_obj = ClassificationObject(accuracy_score(Y_test, predicted)*100
                                    , time.time() - knn_start_time)

    return knn_obj


def DTClassifier(df):
    tree_start_time = time.time()
    X = df.drop(['classe'], axis=1)
    Y = df['classe']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=None)

    dtc = DecisionTreeClassifier(max_depth=7).fit(X_train, Y_train)

    predicted = dtc.predict(X_test)

    dtc_obj = ClassificationObject(accuracy_score(Y_test, predicted)*100, 
                                    time.time() - tree_start_time)
    return dtc_obj


def averageResultClassifier(execution_obj):
    sum = 0
    for r in execution_obj:
        sum += r.score

    totalAverage = sum/len(execution_obj)

    return totalAverage

def averageExecutionTime(execution_obj):
    sum = 0
    for r in execution_obj:
        sum += r.exec_time

    totalAverage = sum/len(execution_obj)
    return totalAverage

dataframe = pd.read_csv("car_evaluation.csv")

dataframe.columns = ['preco', 'manutencao', 'portas',
                     'capacidade', 'p_malas', 'seguranca', 'classe']

#análise do formato dos dados que recebemos

print(dataframe.info(), "\n")

#pré-análise de quais classes são mais recorrentes no ds
print("\n\n", dataframe['classe'].value_counts())

sns.barplot(x=dataframe['classe'].value_counts(), y=dataframe['classe'].value_counts().index)
plt.xlabel('Frequencia')
plt.ylabel('Classe')
plt.show()

#passando os dados para ordinalEncoding a fim de conseguir usar a função fit de cada metodo 
#de classificacao

ordinalEncoder = ce.OrdinalEncoder(cols=['preco', 'manutencao', 'portas',
                     'capacidade', 'p_malas', 'seguranca', 'classe'])

dataframe = ordinalEncoder.fit_transform(dataframe)

print("\n\n", dataframe.head)

knnexecutionVet = []
dtcexecutionVet = []
rdf10executionVet = []
rdf100executionVet = []

#100 execuções de cada método de classificação

for n in range(100):
    knnexecutionVet.append(KNNClassifier(dataframe))
    rdf10executionVet.append(RDFClassifier(dataframe, 10))
    rdf100executionVet.append(RDFClassifier(dataframe, 100))
    dtcexecutionVet.append(DTClassifier(dataframe))

knn_avg = averageResultClassifier(knnexecutionVet)
rdf10_avg = averageResultClassifier(rdf10executionVet)
rdf100_avg = averageResultClassifier(rdf100executionVet)
dtc_avg = averageResultClassifier(dtcexecutionVet)

print("\nAverage KNN: ", knn_avg)
print("\nAverage Random Forest 10: ", rdf10_avg)
print("\nAverage Random Forest 100: ", rdf100_avg)
print("\nAverage Decision Tree: ", dtc_avg)

knn_time = averageExecutionTime(knnexecutionVet)
rdf10_time = averageExecutionTime(rdf10executionVet)
rdf100_time = averageExecutionTime(rdf100executionVet)
dtc_time = averageExecutionTime(dtcexecutionVet)

print("\nTime KNN: ", knn_time)
print("\nTime Random Forest 10: ", rdf10_time)
print("\nTime Random Forest 100: ", rdf100_time)
print("\nTime Decision Tree: ", dtc_time)


sns.barplot(x = ['KNN', 'RF10', 'RF100', 'D Tree'], 
            y=[knn_avg, rdf10_avg, rdf100_avg, dtc_avg])
plt.ylabel("Média Predições")
plt.show()

sns.barplot(x = ['KNN', 'RF10', 'RF100', 'D Tree'], 
            y=[knn_time, rdf10_time, rdf100_time, dtc_time])
plt.ylabel("Segundos")
plt.show()

print("\n\nExecution time: %s" % (time.time() - start_time))