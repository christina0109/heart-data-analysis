from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import RandomizedLogisticRegression as RLR
from sklearn.metrics import roc_auc_score,precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

def model(model,X_train,X_test,y_train,y_test,model_name):
    """ 第一个结果为模型 ， 第二个正确率，
    三 AUC值，四 预测结果，五预测概率"""
    logis_reg = model(random_state=10)
    logis_reg.fit(X_train, y_train)
    predtest=logis_reg.predict(X_test)
    probtest=logis_reg.predict_proba(X_test)[:,1]
    probtrain= logis_reg.predict_proba(X_train)[:,1]
    print(model_name)
#     print(u'模型的训练集平均正确率为：%s' % logis_reg.score(X_train, y_train))
#     print(u'模型的测试集平均正确率为：%s' % logis_reg.score(X_test, y_test))
    #模型评估 AUC 的值
    score=[logis_reg.score(X_train, y_train),logis_reg.score(X_test, y_test)]
#     print(' AUC (Train)',roc_auc_score(y_train,probtrain_lr))
#     print(' AUC (Test)',roc_auc_score(y_test,probtest_lr))
    auc=[roc_auc_score(y_train,probtrain),roc_auc_score(y_test,probtest)]
    return logis_reg,score,auc,predtest,probtest
    
    
 logisl2=model(LR,X_train,X_test,y_train,y_test,model_name='L2正正则化 LR模型：') 
 logisl2[1][1]
 
 # 评判结果报告
 from sklearn.metrics import classification_report
 def my_classification_report(y_true, y_pred):
    from sklearn.metrics import classification_report
    print( "classification_report(left: labels):")
    print (classification_report(y_true, y_pred))
    
 my_classification_report(y_test,y_score_pre)


#筛选特征RLR
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import RandomizedLogisticRegression as RLR

Rlogis_reg = RLR(random_state=10) #筛选变量
Rlogis_reg.fit(X_train, y_train)
selected_col =list(Rlogis_reg.get_support())
selected_col.append(False)
print(u"通过随机逻辑回归模型筛选特征结束")
print(u"有效特征为：%s" % ",".join(smote_resampled.columns[selected_col]))

X1_train = pd.DataFrame(X_train,columns=names)
X1_test = pd.DataFrame(X_test,columns=names)
selected_feature =smote_resampled.columns[selected_col]
X1_train = X1_train[selected_feature]
X1_test = X1_test[selected_feature] # 筛选好特征
RLR=model(LR,X1_train,X1_test,y_train,y_test,model_name='RLR 个4 特征：') 


#决策树
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier as DTC

dtc=model(DTC,X_train,X_test,y_train,y_test,model_name='决策树') 

### 2-3 决策树筛选特征
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.feature_selection import RFE

def sele_feature(model,n):
    clf = model(random_state=10) 
    rfe = RFE(clf,n_features_to_select=n,step = 1)
    rfe.fit(X_train,y_train)
    selector=rfe.fit(X_train,y_train)
    print("特征排序:")
    print (sorted(zip(map(lambda x: round(x, 4), selector.ranking_), names)))
    return selector

#随机森林
from sklearn.ensemble import RandomForestClassifier as RFC
rfc=model(RFC,X_train,X_test,y_train,y_test,model_name='随机森林模型：') 


#网格搜索
from sklearn.model_selection import GridSearchCV

def rf_cross_validation(Model,X_train, y_train):        
    model = Model(random_state=10)
    param_grid = {'n_estimators':range(5,50,2),'max_depth':range(1,6,1),'min_samples_split':range(2,5)}    
    grid_search = GridSearchCV(model, param_grid, n_jobs = 8, verbose=1,cv=5)    
    grid_search.fit(X_train, y_train) 
    best_parameters = grid_search.best_estimator_.get_params()    
#     for para, val in list(best_parameters.items()):    
#         print(para, val)    
    model = Model( n_estimators=best_parameters['n_estimators'],
                                   max_depth=best_parameters['max_depth'], 
                                   min_samples_split=best_parameters['min_samples_split'],
                                   random_state=10)                                      
#     model.fit(X_train, y_train)  
    best_params = print('最佳参数：%s'% grid_search.best_params_ )
    best_score = print('最佳得分：%s'% grid_search.best_score_ )
    return model,best_params,best_score
    
def model_1(model,X_train,X_test,y_train,y_test,model_name):
    model.fit(X_train, y_train)
    predtest_lr=model.predict(X_test)
    probtest_lr=model.predict_proba(X_test)[:,1]
    predtrain_lr = model.predict(X_train)
    probtrain_lr = model.predict_proba(X_train)[:,1]
    score=[model.score(X_train, y_train),model.score(X_test, y_test)]
    auc=[roc_auc_score(y_train,probtrain_lr),roc_auc_score(y_test,probtest_lr)]
    return model,score,auc    
    
param=rf_cross_validation(RFC,X_train, y_train)
model_1(param[0],X_train,X_test,y_train,y_test,model_name='randomTRee')

SVM
from sklearn.svm import SVC

def model_svc(model,X_train,X_test,y_train,y_test,model_name):

    logis_reg = model(random_state=10,probability=True,class_weight='balanced')
    logis_reg.fit(X_train, y_train)
    predtest_lr=logis_reg.predict(X_test)
    probtest_lr=logis_reg.predict_proba(X_test)[:,1]
    predtrain_lr = logis_reg.predict(X_train)
    probtrain_lr = logis_reg.predict_proba(X_train)[:,1]

#     print(model_name)
#     print(u'模型的训练集平均正确率为：%s' % logis_reg.score(X_train, y_train))
#     print(u'模型的测试集平均正确率为：%s' % logis_reg.score(X_test, y_test))
#     #模型评估 AUC 的值
    score=[logis_reg.score(X_train, y_train),logis_reg.score(X_test, y_test)]
#     print(' AUC (Train)',roc_auc_score(y_train,probtrain_lr))
#     print(' AUC (Test)',roc_auc_score(y_test,probtest_lr))
    auc=[roc_auc_score(y_train,probtrain_lr),roc_auc_score(y_test,probtest_lr)]
    return logis_reg,score,auc
     
svc=model_svc(SVC,X_train,X_test,y_train,y_test,model_name='SVM')


XGBOOSt
import xgboost as xgb
dtrain=xgb.DMatrix(X_train,label=y_train)
dtest=xgb.DMatrix(X_test)

params={'booster':'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth':7,
        'lambda':15,
        'subsample':0.75,
        'colsample_bytree':0.75,
        'min_child_weight':1,
        'eta': 0.02,
        'seed':0,
        'silent':1,
        'gamma':0.15,
        'learning_rate' : 0.01}
        
bst=xgb.train(params,dtrain,num_boost_round=800)
ypred=bst.predict(dtest)

y_pred = (ypred >= 0.5)*1

from sklearn import metrics
print ('AUC: %.4f' % metrics.roc_auc_score(y_test,ypred))
# print ('Accuray: %.4f' % bst.score( y_test,y_pred))
print ('Recall: %.4f' % metrics.recall_score(y_test,y_pred))
print ('F1-score: %.4f' %metrics.f1_score(y_test,y_pred))
print ('Precesion: %.4f' %metrics.precision_score(y_test,y_pred))
metrics.confusion_matrix(y_test,y_pred)

#adaboost
from sklearn.metrics import zero_one_loss
from sklearn.ensemble import AdaBoostClassifier as Adab  

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss
from sklearn.ensemble import AdaBoostClassifier
import time
a=time.time()

n_estimators=400
learning_rate=1.0

#决策树桩
dt_stump=DecisionTreeClassifier(max_depth=5,min_samples_leaf=2)
dt_stump.fit(X_train,y_train)
dt_stump_err=1.0-dt_stump.score(X_test,y_test)

#决策树桩的生成
ada_discrete=AdaBoostClassifier(base_estimator=dt_stump,learning_rate=learning_rate,
                                n_estimators=n_estimators,algorithm='SAMME')
ada_discrete.fit(X_train,y_train)


ada_real=AdaBoostClassifier(base_estimator=dt_stump,learning_rate=learning_rate,
                                n_estimators=n_estimators,algorithm='SAMME.R')#相比于ada_discrete只改变了Algorithm参数
ada_real.fit(X_train,y_train)

fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot([1,n_estimators],[dt_stump_err]*2,'k-',label='Decision Stump Error')
# ax.plot([1,n_estimators],[dt_err]*2,'k--',label='Decision Tree Error')

ada_discrete_err=np.zeros((n_estimators,))
for i,y_pred in enumerate(ada_discrete.staged_predict(X_test)):
    ada_discrete_err[i]=zero_one_loss(y_pred,y_test)#0-1损失，类似于指示函数
ada_discrete_err_train=np.zeros((n_estimators,))
for i,y_pred in enumerate(ada_discrete.staged_predict(X_train)):
    ada_discrete_err_train[i]=zero_one_loss(y_pred,y_train)

ada_real_err=np.zeros((n_estimators,))
for i,y_pred in enumerate(ada_real.staged_predict(X_test)):
    ada_real_err[i]=zero_one_loss(y_pred,y_test)
ada_real_err_train=np.zeros((n_estimators,))
for i,y_pred in enumerate(ada_real.staged_predict(X_train)):
    ada_discrete_err_train[i]=zero_one_loss(y_pred,y_train)


ax.plot(np.arange(n_estimators)+1,ada_discrete_err,label='Discrete AdaBoost Test Error',color='red')
ax.plot(np.arange(n_estimators)+1,ada_discrete_err_train,label='Discrete AdaBoost Train Error',color='blue')
ax.plot(np.arange(n_estimators)+1,ada_real_err,label='Real AdaBoost Test Error',color='orange')
ax.plot(np.arange(n_estimators)+1,ada_real_err_train,label='Real AdaBoost Train Error',color='green')


ax.set_ylim((0.0,0.5))
ax.set_xlabel('n_estimators')
ax.set_ylabel('error rate')






