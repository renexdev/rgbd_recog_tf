import matplotlib.pyplot as plt
import numpy as np
import sys, os, ipdb
from sklearn.metrics import confusion_matrix

#EXPERIMENT = 'fuseval_299'

def main(pth):
    # load data
    data = np.loadtxt(pth)
    N = len(data)
    n_classes = 51
    score = data[:,:n_classes]
    y_pred = data[:,n_classes].astype(np.int32)
    y_true = data[:,n_classes+1].astype(np.int32)

    y_true_arr = np.zeros((N,n_classes))
    for i in range(N): 
        y_true_arr[i,y_true[i]] = 1

    # compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # normalize
    print 'Prediction classes\n', np.unique(y_pred)
    print 'True classes\n', np.unique(y_true)

    # compute accuracy on top N
    for t in range(5):
        count = 0
        for i in range(N):
            foo = score[i]
            if y_true[i] in foo.argsort()[-(t+1):][::-1]: count+=1
        print 'Top', t+1, 'accuracy:', count*1.0/N

    # plot
    plt.figure()
    plt.imshow(cm)
    plt.colorbar()
    plt.title('Confusion matrix: ')

    plt.figure()
    ic = [cm[i,i] for i in range(n_classes)] # in-class accuracy
    plt.bar(range(n_classes), ic, 0.5)
    plt.grid('on')
    plt.title('In-class accuracy')


if __name__ == '__main__':
    for pth in sys.argv[1:]:
        main(pth)
    plt.show()
