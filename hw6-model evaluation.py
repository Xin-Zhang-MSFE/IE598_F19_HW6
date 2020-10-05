import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score,GridSearchCV
import time
from sklearn.metrics import classification_report

df=pd.read_csv('ccdefault.csv',index_col='ID')
df.head()

df.dropna()

X=df.iloc[:,0:23].values
y=df.iloc[:,23].values
#part1 test the random_state
tree=DecisionTreeClassifier(criterion='gini',max_depth=5)
randomrange=range(1,11)
inscore=[]
outscore=[]
start = time.perf_counter()
for i in randomrange:
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=i)
    tree.fit(X_train,y_train)
    y_pred_train=tree.predict(X_train)
    y_pred_test=tree.predict(X_test)
    outscore.append(accuracy_score(y_test, y_pred_test))
    inscore.append(accuracy_score(y_train, y_pred_train))
    
print("In-sample accuracy score:")
print([float('{:.4f}'.format(i)) for i in inscore])
print('Mean of In-sample: %.4f, Std: %.4f' %(np.mean(inscore),np.std(inscore)))
print("Out-sample accuracy score:")
print([float('{:.4f}'.format(i)) for i in outscore])
print('Mean of Out-sample: %.4f, Std: %.4f' %(np.mean(outscore),np.std(outscore)))
end = time.perf_counter()
print ("Using time:",end-start,"s")
plt.plot(randomrange,inscore,color='black',label = 'Testing Accuracy')
plt.plot(randomrange,outscore,color='red',label = 'Training Accuracy')
plt.xlabel('random_state')
plt.ylabel('accuracy score')
plt.legend()
plt.ylim(0.75, 0.85)
plt.show()

#part2  Cross validation
start = time.perf_counter()
cv_score=cross_val_score(DecisionTreeClassifier(), X_test, y_test,cv=10)
print("Out-sample CV accuracy score:")
print([float('{:.4f}'.format(i)) for i in cv_score])
print('Mean of Out-sample: %.4f, Std: %.4f' %(np.mean(cv_score),np.std(cv_score)))
end = time.perf_counter()
print ("Using time:",end-start,"s")

#gridsearch
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=42)
tree=DecisionTreeClassifier()
param={'criterion':('gini','entropy'),'max_depth':range(5,15)}
cv=GridSearchCV(tree,param_grid=param)
cv.fit(X_train,y_train)
y_pred=cv.predict(X_test)
print('')
print('GridSearch:')
print('Accuracy:%.4f'% cv.score(X_test,y_test))
print('Tuned Model Parameters:{}'.format(cv.best_params_))
#print(classification_report(y_test, y_pred))

print("My name is Xin Zhang")
print("My NetID is: xzhan81")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")









