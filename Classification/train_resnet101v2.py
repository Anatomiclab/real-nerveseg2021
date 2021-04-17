import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import keras
import keras.backend as K
import click
import logging
import csv

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc

#define parameters
BATCH_SIZE = 12
IMG_SIZE = (224, 224)
dp_rate = 0.3
base_learning_rate=0.001
initial_epochs = 10
fine_tune_epochs = 15
fine_tune_at = 360
folder = "resnet101v2"


PATH=os.getcwd()

train_dir = os.path.join(PATH, 'datasets/101611271211/train')
validation_dir = os.path.join(PATH, 'datasets/101611271211/val')
all_dir = os.path.join(PATH, 'datasets/101611271211/all')

train_dataset = image_dataset_from_directory(all_dir,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
											 validation_split=.2,
											 subset="training",
											 seed = 1337,
                                             image_size=IMG_SIZE)

validation_dataset = image_dataset_from_directory(all_dir,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
												  validation_split=.2,
												  subset="validation",
												  seed = 1337,
                                                  image_size=IMG_SIZE)


												  
print("train dataset: {}" .format(train_dataset))
print("validation dataset: {}" .format(validation_dataset))


print(train_dataset.class_names)
print(validation_dataset.class_names)
class_names = train_dataset.class_names
				
for image_batch, labels_batch in train_dataset:
  print(image_batch.shape)
  print(labels_batch.shape)
  break
				

#<TMP>Rescaling pixel for adpating to the pretrained model
preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

# Create the base model from the pre-trained model MobileNet V2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.ResNet101V2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')


data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

 
image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

base_model.trainable = False

e = open(folder+"/model.txt", 'a')
print("summary: {}" .format(base_model.summary()), file=e)
print("Depth of the model is: {}" .format(len(base_model.layers)), file=e)


global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(3, activation='softmax')
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

#inputs = tf.keras.Input(shape=(224, 224, 3))
inputs = tf.keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(dp_rate)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

	

#normalized_ds = train_dataset.map(lambda x, y: (normalization_layer(x), y))
#image_batch, labels_batch = next(iter(normalized_ds))
#first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
#print(np.min(first_image), np.max(first_image)) 




# compile and train the model

model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])



print("summary: {}" .format(model.summary()), file=e)
print("Depth of the model is: {}" .format(len(model.layers)), file=e)
e.close()



history = model.fit(train_dataset, epochs=initial_epochs, 
                    validation_data=validation_dataset)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig(folder+"/acc_loss.png")




##############################################Start fine tuning#######################################
	

base_model.trainable = True

for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate/10),
              metrics=['accuracy'])

model.summary()

e = open(folder+"/model.txt", 'a')
print(len(model.trainable_variables))
print("Depth of the model is: {}" .format(len(model.trainable_variables)), file=e)
e.close()


total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(train_dataset,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=validation_dataset)
	
	
acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']
loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig(folder+"/acc_loss_after_fine_tuning.png")



	
########################################################SAVE fine tuned weighting#############################
tf.keras.models.save_model(model, filepath=folder+'/saved_model/current') 






























#all_dir = os.path.join(PATH, 'datasets/101611271211/all')


###############################################Evaluate#######################################################

def eva(TV):
	if TV == "train":
		z1=os.path.join(PATH, folder, folder+"_train_CM.txt")
		z2=os.path.join(PATH, folder, folder+"_train_F1.txt")
		z3=os.path.join(PATH, folder, folder+"_train_AUC.txt")
		z4=os.path.join(PATH, folder, folder+"_train_roc.png")
		all_dataset = train_dataset
	if TV == "val":
		z1=os.path.join(PATH, folder, folder+"_val_CM.txt")
		z2=os.path.join(PATH, folder, folder+"_val_F1.txt")
		z3=os.path.join(PATH, folder, folder+"_val_AUC.txt")
		z4=os.path.join(PATH, folder, folder+"_val_roc.png")
		all_dataset = validation_dataset
	
	print("Generate predictions for all data for {}" .format(TV))
    ##############################
	prediction_score_for_ROC = []
	prediction_label_for_ROC = []
	prediction_flattened=[]
	prediction_true_label=[]
	
	all_dataset_iterator =  all_dataset.as_numpy_iterator()
	
	for k in range(tf.data.experimental.cardinality(all_dataset)):
		image_batch, label_batch = all_dataset_iterator.next()
		predictions = model.predict_on_batch(image_batch)
		prediction_true_label = [*prediction_true_label,*label_batch]
        #print("prediction_true_label:",prediction_true_label)
		for i in range(len(predictions)):
			_zeroArray = np.zeros(3)
			_zeroArray[label_batch[i]] = 1
			prediction_label_for_ROC.append(_zeroArray)
			predictions_softmax = tf.nn.softmax(predictions)
			prediction_score_for_ROC.append(predictions_softmax[i])
			#print("index:{}, array:{}, max index:{}".format(i,predictions[i],np.argmax(predictions[i])))
			prediction_flattened.append( np.argmax(predictions[i]))
            ######modify here
            #if i == bs:
                #if label_batch[bs] != np.argmax(predictions[bs]):
                    #tfs = "Wrong"
                #else:
                    #tfs = "Correct"
                #title =  tfs +": True class:"+class_names[label_batch[bs]]+" ,Predicted class:"+class_names[np.argmax(predictions[bs])]
                #PN = os.path.join(rPATH, Mname+nos, Mname+nos+tfs+class_names[label_batch[bs]]+class_names[np.argmax(predictions[bs])]+str(k)+str(bs)+".png")
                #plt.imshow(image_batch[bs].astype("uint8"))
                #plt.title(title)
                #plt.axis("off")
                #plt.savefig(PN)
			print("Finished predicting {}batches({}images)" .format(k, k*BATCH_SIZE))
			#every wrong plot
            #if label_batch[i] != np.argmax(predictions[i]):
                #title = "True class:"+class_names[label_batch[i]]+" ,Predicted class:"+class_names[np.argmax(predictions[i])]
                #PN = os.path.join(rPATH, Mname+nos, Mname+nos+"False"+class_names[label_batch[i]]+class_names[np.argmax(predictions[i])]+str(k)+str(i)+".png")
                #plt.imshow(image_batch[i].astype("uint8"))
                #plt.title(title)
                #plt.axis("off") 
                #plt.savefig(PN)
    
    
	predictionsA = np.array(prediction_flattened)
	prediction_true_label = np.array(prediction_true_label)
	prediction_score_for_ROC = np.array(prediction_score_for_ROC)
	prediction_label_for_ROC = np.array(prediction_label_for_ROC)
	s = open(z1, 'a')
	print(tf.math.confusion_matrix(prediction_true_label, predictionsA), file=s)
	s.close()
	q = open(z2, 'a')
	import sklearn
	from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
	print('\nAccuracy: {:.4f}\n'.format(accuracy_score(prediction_true_label, predictionsA)), file=q)
    
	print('Micro Precision: {:.4f}'.format(precision_score(prediction_true_label, predictionsA, average='micro')), file=q)
	print('Micro Recall: {:.4f}'.format(recall_score(prediction_true_label, predictionsA, average='micro')), file=q)
	print('Micro F1-score: {:.4f}\n'.format(f1_score(prediction_true_label, predictionsA, average='micro')), file=q)
    
	print('Macro Precision: {:.4f}'.format(precision_score(prediction_true_label, predictionsA, average='macro')), file=q)
	print('Macro Recall: {:.4f}'.format(recall_score(prediction_true_label, predictionsA, average='macro')), file=q)
	print('Macro F1-score: {:.4f}\n'.format(f1_score(prediction_true_label, predictionsA, average='macro')), file=q)
    
	print('Weighted Precision: {:.4f}'.format(precision_score(prediction_true_label, predictionsA, average='weighted')), file=q)
	print('Weighted Recall: {:.4f}'.format(recall_score(prediction_true_label, predictionsA, average='weighted')), file=q)
	print('Weighted F1-score: {:.4f}'.format(f1_score(prediction_true_label, predictionsA, average='weighted')), file=q)
    
	from sklearn.metrics import classification_report
	print('\nClassification Report\n', file=q)
	print(classification_report(prediction_true_label, predictionsA, target_names=['nerve', 'opening wound', 'tendon']), file=q)
    
	from sklearn.metrics import multilabel_confusion_matrix
	from sklearn.metrics import matthews_corrcoef
	MCC = matthews_corrcoef(prediction_true_label, predictionsA)
	print("Matthew's coefficient is: ", file=q)
	print(MCC, file=q)
	print("	", file=q)
	mcm = multilabel_confusion_matrix(prediction_true_label, predictionsA)
    #print("mcm is: \n", file=q)
    #print(mcm, file=q)
	tn = mcm[:, 0, 0]
	tp = mcm[:, 1, 1]
	fn = mcm[:, 1, 0]
	fp = mcm[:, 0, 1]
	print("True positive:", file=q)
	print(tp, file=q)
	print("	", file=q)
	print("True negative:", file=q)
	print(tn, file=q)
	print("	", file=q)
	print("False positive:", file=q)
	print(fp, file=q)
	print("	", file=q)
	print("False negative:", file=q)
	print(fn, file=q)
	sensitivity = tp / (tp + fn)
	precision = tp / (tp + fp)
	specificity = tn / (tn + fp)
	NPV = tn / (tn + fn)
	accuracy3 = (tp + tn)/ (tp + fp + tn + fn)
    
	from math import sqrt
	print("Per class Matthews coefficient:	", file=q)
	for i in [0,1,2]:
		MCC1 = (tp[i]*tn[i] - fp[i]*fn[i])/sqrt((tp[i]+fp[i])*(tp[i]+fn[i])*(tn[i]+fp[i])*(tn[i]+fn[i]))
		print(MCC1, file=q)
	print("	", file=q)
    
	print("\n", file=q)
	print("nerve	opening wound	tendon\n", file=q)
	print("Sensitivity, hit rate, recall, or true positive rate:	", file=q)
	print(sensitivity, file=q)
	print("	", file=q)
	print("Specificity or true negative rate:	", file=q)
	print(specificity, file=q)
	print("	", file=q)
	print("precision or positive predictive value:	", file=q)
	print(precision, file=q)
	print("	", file=q)
	print("Negative predictive value:	", file=q)
	print(NPV, file=q)
	print("	", file=q)
	print("Accuracy:	", file=q)
	print(accuracy3, file=q)
	print("	", file=q)
	F1P = []
	for i in [0,1,2]:
		F1P.append((2*precision[i]*sensitivity[i])/(precision[i]+sensitivity[i]))
    
	F1N = []
	for i in [0,1,2]:
		F1N.append((2*NPV[i]*specificity[i])/(NPV[i]+specificity[i]))
    
	print("F1 score positive(2*precision*specificity)/(precision+sensitivity)", file=q)
	print(F1P, file=q)
	print("	", file=q)
	print("F1 score negative(2*NPV*specificity)/(NPV+specificity)", file=q)
	print(F1N, file=q)
	print("	", file=q)
	print("\n", file=q)
	print("\n", file=q)
	print("Macro average metrics below\n", file=q)
	print("\n", file=q)
    
	Msensitivity=0
	Mspecificity=0
	Mprecision=0
	MNPV=0
	Maccuracy3=0
	MF1P=0
	MF1N=0
	for i in [0,1,2]:
		Msensitivity=Msensitivity+sensitivity[i]
	Msensitivity=Msensitivity/3
	for i in [0,1,2]:
		Mspecificity=Mspecificity+specificity[i]
	Mspecificity=Mspecificity/3
	for i in [0,1,2]:
		Mprecision=Mprecision+precision[i]
	Mprecision=Mprecision/3
	for i in [0,1,2]:
		MNPV=MNPV+NPV[i]
	MNPV=MNPV/3
	for i in [0,1,2]:
		Maccuracy3=Maccuracy3+accuracy3[i]
	Maccuracy3=Maccuracy3/3
	for i in [0,1,2]:
		MF1P=MF1P+F1P[i]
	MF1P=MF1P/3
	for i in [0,1,2]:
		MF1N=MF1N+F1N[i]
	MF1N=MF1N/3
    
	print("	", file=q)
	print("Macro Sensitivity, hit rate, recall, or true positive rate:	", file=q)
	print(Msensitivity, file=q)
	print("	", file=q)
	print("Macro Specificity or true negative rate:	", file=q)
	print(Mspecificity, file=q)
	print("	", file=q)
	print("Macro precision or positive predictive value:	", file=q)
	print(Mprecision, file=q)
	print("	", file=q)
	print("Macro Negative predictive value:	", file=q)
	print(MNPV, file=q)
	print("	", file=q)
	print("Macro Accuracy:	", file=q)
	print(Maccuracy3, file=q)
	print("	", file=q)
	print("Macro F1 score positive(2*precision*specificity)/(precision+sensitivity)", file=q)
	print(MF1P, file=q)
	print("	", file=q)
	print("Macro F1 score negative(2*NPV*specificity)/(NPV+specificity)", file=q)
	print(MF1N, file=q)
	print("	", file=q)
    
	print("\n", file=q)
	print("\n", file=q)
	print("Micro average metrics below\n", file=q)
	print("\n", file=q)
	atp=0
	atn=0
	afp=0
	afn=0
	for i in [0,1,2]:
		atp=atp + tp[i]
	for i in [0,1,2]:
		atn=atn + tn[i]
	for i in [0,1,2]:
		afp=afp + fp[i]	
	for i in [0,1,2]:
		afn=afn + fn[i]
	Asensitivity = atp / (atp + afn)
	Aspecificity = atn / (atn + afp)
	Aprecision = atp / (atp + afp)
	ANPV=atn / (atn + afn)
	Aaccuracy3= (atp + atn)/ (atp + afp + atn + afn)
	AF1P=(2*Aprecision*Asensitivity)/(Aprecision+Asensitivity)
	AF1N=(2*ANPV*Aspecificity)/(ANPV+Aspecificity)
	print("	", file=q)
	print("micro Sensitivity, hit rate, recall, or true positive rate	:\n", file=q)
	print(Asensitivity, file=q)
	print("	", file=q)
	print("micro Specificity or true negative rate:	", file=q)
	print(Aspecificity, file=q)
	print("	", file=q)
	print("micro precision or positive predictive value:	", file=q)
	print(Aprecision, file=q)
	print("	", file=q)
	print("micro Negative predictive value:	", file=q)
	print(ANPV, file=q)
	print("	", file=q)
	print("micro Accuracy:	", file=q)
	print(Aaccuracy3, file=q)
	print("	", file=q)
	print("micro F1 score positive(2*precision*specificity)/(precision+sensitivity):	", file=q)
	print(AF1P, file=q)
	print("	", file=q)
	print("micro F1 score negative(2*NPV*specificity)/(NPV+specificity):	", file=q)
	print(AF1N, file=q)
        
	q.close()
    
#AUC
	import matplotlib
    
    
    # Compute ROC curve and ROC area for each class
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(3):
		fpr[i], tpr[i], _ = roc_curve(prediction_label_for_ROC[:, i], prediction_score_for_ROC[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
	fpr["micro"], tpr["micro"], _ = roc_curve(prediction_label_for_ROC.ravel(), prediction_score_for_ROC.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # First aggregate all false positive rates
	all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
    
    # Then interpolate all ROC curves at this points
	mean_tpr = np.zeros_like(all_fpr)
	for i in range(3):
		mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
	mean_tpr /= 3
    
	fpr["macro"] = all_fpr
	tpr["macro"] = mean_tpr
	roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
	q = open(z3, 'a')
	print(roc_auc, file=q)
        
	q.close()




    # Plot all ROC curves
	from itertools import cycle
	plt.figure(figsize=(16, 16))
    
	fpr["micro"], tpr["micro"], _ = roc_curve(prediction_label_for_ROC.ravel(), prediction_score_for_ROC.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
	plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.4f})'''.format(roc_auc["micro"]),color='deeppink', linestyle=':', linewidth=4)
    
	plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.4f})'''.format(roc_auc["macro"]),color='navy', linestyle=':', linewidth=4)
    
	lw=2
	#class_names=["nerve","open","tendon"]
	colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
	for i, color in zip(range(3), colors):
		plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.4f})'''.format(class_names[i], roc_auc[i]))
    
	plt.plot([0, 1], [0, 1], 'k--', lw=lw)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic to multi-class')
	plt.legend(loc="lower right")
	plt.savefig(z4)

	print("Finished evaluating {}" .format(TV))

#########################
eva("train")
eva("val")


