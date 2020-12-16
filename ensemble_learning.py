import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score,f1_score
from numpy.linalg import norm
from itertools import product


def read_target(filename):
    t = []
    with open (filename) as myfile:                #reading the input points and target labels
        for line in myfile.readlines():
            x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13 = map(float,line.split(','))
            t.append(x13)
    t = np.array(t).reshape(len(t),1)
    return t

def read_labels(filename):
    t = []
    with open(filename) as myfile:
        for i,line in enumerate(myfile):
            for numstr in line.split('\n'):
                if numstr:
                    try:
                        numfl = float(numstr)
                        t.append(numfl)
                    except ValueError as e:
                        print("error",e,"on line",i)
    print(len(t))
    #t = np.array(t).reshape(len(t),1)
    return t
                        

target = read_target('/home/nizwakhan/Documents/Course_5327/project/dataset_testing.txt')
predictor_1 = read_labels('/home/nizwakhan/Documents/Course_5327/project/ensemble-main/predictor_1_y.txt')
predictor_2 = read_labels('/home/nizwakhan/Documents/Course_5327/project/ensemble-main/predictor_2_y.txt')
predictor_3 = read_labels('/home/nizwakhan/Documents/Course_5327/project/ensemble-main/predictor_3_y.txt')

#calculate F1 score and accuracy for each model

y_pred = [predictor_1,predictor_2,predictor_3]

for i,pred in enumerate(y_pred):
    print('standalone accuracy for predictor',i+1,'=',accuracy_score(target,pred))
    print('standalone F1 score for predictor',i+1,'=',f1_score(target,pred))



def majority_voting(yhats,testy):
    #hard voting 
    result,_ = np.array(stats.mode(yhats,axis = 0))
    result = result.reshape(2000,1)
    return accuracy_score(testy, result),f1_score(testy,result)



acc_score,f1 = majority_voting(y_pred, target)
#print('majority voting',acc_score,f1)



def weighted_majority_voting(yhats, weights,target):
    yhats = np.array(yhats).reshape(3,2000)
    # weighted sum across ensemble members
    summed = np.tensordot(yhats, weights, axes=((0),(0)))  #taking dot products between weights and predicted labels
    for i,element in enumerate(summed):
        #print(i)
        if element<0:
            summed[i] = -1
        else:
            summed[i] = 1
    summed = np.expand_dims(summed,axis=1)
    return accuracy_score(target,summed),f1_score(target,summed)



def normalize(weights):
	result = norm(weights, 1)
	if result == 0.0:
		return weights
	return weights / result

n_members = 3
weights = [1.0/n_members for _ in range(n_members)]
acc_score_equal,f1_score_equal = weighted_majority_voting(y_pred,weights,target)
print('Assigning equal weights to all predictors : Accuracy',acc_score_equal,' and F1 score',f1_score_equal)


def grid_search(members,testy):
    w = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    best_score, best_weights = 0.0, None
    for weights in product(w, repeat=len(y_pred)):
    	# skip if all weights are equal
        if len(set(weights)) == 1:
            continue
        weights = normalize(weights)
        acc_score,f1_score = weighted_majority_voting(y_pred,weights,target)
        if f1_score > best_score:
            best_score, best_weights = f1_score, weights
            print('>%s %.3f' % (best_weights, best_score))
    return list(best_weights)

weights = grid_search(n_members,y_pred)
acc_best_score,f1_best_score = weighted_majority_voting(y_pred, weights,target)
print('After grid search the optimal weights found were ',weights,'which gives us accuracy = ', acc_best_score,'and F1 score =',f1_best_score)





