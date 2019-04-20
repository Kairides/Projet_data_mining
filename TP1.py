import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mca2 import *
# import graphviz

# import seaborn as sns
# from sklearn.model_selection import train_test_split

# from sklearn.linear_model import LinearRegression
from sklearn import tree
import graphviz
from graphviz import Digraph

dot = Digraph(comment='haha, benis')

# from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error
df = pd.read_csv('german.data', sep='\t')

# print('dtype: ', data.dtypes)
# print('shape: ', data.shape)
# print('count: ', data.count())
# print(data.describe().max)
'''
spread = data.quantile(.5)
center = data.quantile(.5)
flier_high = data.quantile(.9)
flier_low = data.quantile(.1)
bp = np.concatenate((spread, center, flier_high, flier_low), 0)

plt.boxplot(bp)

plt.show()
duration	credit_history	purpose	credit_amount	savings_status	employment	installment_commitment	personal_status	other_parties	residence_since	property_magnitude	age	other_payment_plans	housing	existing_credits	job	num_dependents	own_telephone	foreign_worker	class
'''
# df.describe()
# df.duration.plot.box()
# plt.show()

# df.purpose.value_counts().plot.pie(figsize=[5,5])
# plt.show()

dfbin = pd.get_dummies(df.iloc[:,:20])
# print(dfbin.describe())

dfmca = mca(dfbin, benzecri=False)
nc = dfmca.fs_r(N=61)
# nc.shape -> (1000, 45) -> passe de 61 à 45 variables car certaines sont redondantes

# print(nc)

pd.DataFrame(nc)

# print(df)
classe = df.iloc[:,-1]
# print(df)

# algo = tree.DecisionTreeClassifier(max_depth=2) # pré-élagage
algo = tree.DecisionTreeClassifier(max_depth=10)  # pré-élagage
monModele = algo.fit(nc, classe)  # fonction ajustement
print(monModele.score(nc, classe))  # mesure de qualité

data = tree.export_graphviz(algo, '28_stab_wounds.dot', max_depth=100)
graphviz.Source(data)
# http://www.webgraphviz.com/


''' X = data.drop(columns=['title', 'popularity'])
y = pd.DataFrame(data['popularity'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

m = tree.DecisionTreeRegressor()
m.fit(X_train, y_train)
prediction = m.predict(X_test)
# print('Coefficients: \n', m.coef_)
def evaluation(y_test, prediction):
	# The mean squared error
	print("Mean absolute error: %.2f"% mean_absolute_error(y_test, prediction))
	print("Median absolute error: %.2f"% median_absolute_error(y_test, prediction))
	# Explained variance score: 1 is perfect prediction
	print('Variance score: %.2f' % r2_score(y_test, prediction))

evaluation(y_test, prediction)

dot_data = tree.export_graphviz(m, 'benis', max_depth=5)
graph = graphviz.Source(dot_data)
'''
