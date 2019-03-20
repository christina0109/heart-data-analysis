# 画混淆矩阵
from sklearn.metrics import confusion_matrix

def confusion_matrix_plot(y_test,y_predict,names):
	import seaborn as sn
	cm = confusion_matrix(y_test,y_predict)
	df_cm = pd.DataFrame(cm)
	f,ax=plt.subplots(figsize=(10,5))
	sn.heatmap(df_cm,annot=True, fmt="d", ax=ax)
	plt.title(names)
	plt.ylabel('Predicted label')
	plt.xlabel('True label')
	
	
# AUC curve	
from sklearn.metrics import roc_auc_score,precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
def Auc(y_test,y_probtest):
	fpr,tpr,threshold = roc_curve(y_test, y_probtest)
	roc_auc = auc(fpr,tpr)
	plt.figure()
	lw = 2
	plt.figure(figsize=(10,10))
	plt.plot(fpr, tpr, color='darkorange',
	         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic logistl2')
	plt.legend(loc="lower right")
	plt.show()
	
	
