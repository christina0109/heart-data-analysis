画混淆矩阵
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
