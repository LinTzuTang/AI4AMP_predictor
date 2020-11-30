import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.metrics import accuracy_score,accuracy_score,f1_score,matthews_corrcoef,confusion_matrix,roc_curve,auc


path = 'Output/PC_6/'

def show_histroy(df,train_acc='acc',validation_acc='val_acc',train_loss='loss',validation_loss='val_loss', path = 'Output/PC_6/'):
    fig1 = plt.figure(figsize=(15,5))
    gs = gridspec.GridSpec(1, 2) 
    ax1 = fig1.add_subplot(gs[0,0])
    ax2 = fig1.add_subplot(gs[0,1])
    #
    ax1.set_title('Train Accuracy',fontsize = '14' )
    ax2.set_title('Train Loss', fontfamily = 'serif', fontsize = '18' )
    ax1.set_xlabel('Epoch', fontfamily = 'serif', fontsize = '13' )
    ax1.set_ylabel('Acc', fontfamily = 'serif', fontsize = '13' )
    ax2.set_xlabel('Epoch', fontfamily = 'serif', fontsize = '13' )
    ax2.set_ylabel('Loss', fontfamily = 'serif', fontsize = '13' )
    ax1.plot(df['acc'], label = 'train',linewidth=2)
    ax1.plot(df['val_acc'], label = 'validation',linewidth=2)
    ax2.plot(df['loss'], label = 'train',linewidth=2)
    ax2.plot(df['val_loss'], label = 'validation',linewidth=2)
    ax1.legend(['train', 'validation'], loc='upper left')
    ax2.legend(['train', 'validation'], loc='upper left')
    fig1.savefig(path+'history.png')
    #plt.show()

def roc(test_label, labels_score, path = 'Output/PC_6/'):
    fpr, tpr, thresholds = roc_curve(test_label, labels_score, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(path+'roc.png')
    plt.show()

def evalution_metrics(test_label, labels_score, save=False, txt_name=None, path = 'Output/PC_6/'):
    accuracy = accuracy_score(test_label, labels_score.round())
    confusion = confusion_matrix(test_label, labels_score.round())
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    precision = TP / float(TP + FP)
    sensitivity = TP / float(FN + TP)
    specificity = TN / float(TN + FP)
    f1 = f1_score(test_label, labels_score.round())
    mcc = matthews_corrcoef(test_label, labels_score.round())
    # precision TP / (TP + FP)
    # recall: TP / (TP + FN)
    # specificity : TN / (TN + FP)
    # f1: 2 TP / (2 TP + FP + FN)
    if save:
        with open(path+'%s_metrics.txt'%txt_name, 'w') as f:
            print('  # Accuracy: %f' % accuracy+'\n', file = f)
            print('  # Precision: %f' % precision+'\n', file = f)  
            print('  # Sensitivity/Recall: %f' % sensitivity+'\n', file = f)
            print('  # Specificity: %f' %specificity+'\n', file = f)
            print('  # F1 score: %f' % f1+'\n', file = f)
            print('  # Matthews Corrcoef:%f' % mcc+'\n', file = f)
        with open(path+'%s_metrics.txt'%txt_name, 'r') as f:    
            for line in f:
                print(line, end="")
    else:
        print('  # Accuracy: %f' % accuracy+'\n')
        print('  # Precision: %f' % precision+'\n')  
        print('  # Sensitivity/Recall: %f' % sensitivity+'\n')
        print('  # Specificity: %f' %specificity+'\n')
        print('  # F1 score: %f' % f1+'\n')
        print('  # Matthews Corrcoef:%f' % mcc+'\n')
        
def metric_array(test_data, test_labels, model):
    labels_score = model.predict(test_data)
    accuracy = accuracy_score(test_labels, labels_score.round())
    confusion = confusion_matrix(test_labels, labels_score.round())
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    precision = TP / float(TP + FP)
    sensitivity = TP / float(FN + TP)
    specificity = TN / float(TN + FP)
    f1 = f1_score(test_labels, labels_score.round())
    mcc = matthews_corrcoef(test_labels, labels_score.round()) 
    metric = [accuracy,precision,sensitivity,specificity,f1,mcc]
    return metric