import random
import math
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix,classification_report

def model_score(x_train, y_train, x_test, y_test, classes = 'multi'):
    
    #x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    model = DecisionTreeClassifier(random_state = 101)
    #print(type(model))
    model.fit(x_train,y_train)
    
    test_predictions = model.predict(x_test)
    if classes == 'multi':
        return f1_score(y_test, test_predictions, average = 'macro')
    else:
        return f1_score(y_test, test_predictions)
    
    
def obj_func(pos_array,X_train,y_train,X_test, y_test, classes):
    alpha = 0.6
    feature_array = np.zeros_like(pos_array)
    for i in range(len(pos_array)):
        #print(1/(1+math.exp(-pos_array[i])))
        feature_array[i] = 1/(1+math.exp(-pos_array[i]))
    feature_array = [round(i) for i in feature_array]
    feature_index = [i for i in range(len(feature_array)) if feature_array[i] > 0]
    #print(len(feature_index))
    if (len(feature_index)==0):
        return [0,0]
    
    f1_score = model_score(X_train, y_train, X_test, y_test, classes = classes)
    fitness_score = alpha*f1_score + (1-alpha)*(X_train.shape[1]-len(feature_index))/X_train.shape[1]
    return [fitness_score,f1_score]



class solution:
    def __init__(self):
        self.best = 0
        self.bestIndividual=[]
        self.convergence = []
        self.optimizer=""
        self.objfname=""
        
        
def SOA(X_train, y_train, X_test, y_test, objf = obj_func, classes = 'multi', lb = -100, ub = 100,
        SearchAgents_no = 100, Max_iter = 100):

    dim = X_train.shape[1]        
    Leader_pos=np.zeros(dim)
    Leader_score=0
    Leader_f1 = 0  
 
    Positions = np.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = np.random.uniform(0,1,SearchAgents_no) *(ub-lb)+lb
    
    convergence_curve=np.zeros(Max_iter)
    s=solution()
    
    t=0
    C_f = 2.0   # fc
    while t<Max_iter:
        for i in range(0,SearchAgents_no):
            for j in range(dim):        
                Positions[i,j]=np.clip(Positions[i,j], lb, ub)
            
            fitness,f1score=objf(Positions[i,:],X_train,y_train,X_test,y_test, classes = classes)
     
            if fitness>Leader_score: 
                Leader_score=fitness; 
                Leader_pos=Positions[i,:].copy()
                Leader_f1= f1score  

        for i in range(0,SearchAgents_no):
            
            A=C_f - t * (C_f/Max_iter) # A=fc−(x×(fc/Maxiteration)) A is employed
                                                 # for the calculation of new search agent position
            B= 2*A*A*(random.uniform(0,1))   # B=2×A2×rd The behavior of Bis randomized which is responsible for proper balancingbetween exploration and exploitation
            ###
            
            C_sp = A * Leader_pos # Cs=A×⃗Ps(x) , ⃗Ps represents the current position of search agent, x indicates the current iteration
            M_sp = B * ( fitness - Leader_pos ) # Ms=B×(⃗Pbs(x)−⃗Ps(x))
            D_sp = C_sp + M_sp  # Ds= | ⃗Cs+⃗Ms |

            ### Based on Eq. 10, 11, 12, 13, 14
            r = np.exp(np.random.uniform(0, 2*np.pi)) # r←u×e^kv ......... u=1, v=1, k←Rand(0,2π)
            temp = r * r* r * (np.sin(np.random.uniform(0, 2*np.pi))) * (np.cos(np.random.uniform(0, 2*np.pi))) * (np.random.uniform(0, 2*np.pi)) # P←x′×y′×z′  
            
            Positions[i,:] = ( D_sp * temp ) + fitness
                    
        convergence_curve[t]=Leader_score
        t=t+1
    
    feature_array = np.zeros_like(Leader_pos)
    for i in range(len(Leader_pos)):
        feature_array[i] = 1/(1+math.exp(-Leader_pos[i]))
    feature_array = [round(i) for i in feature_array]
    feature_index = [i for i in range(len(feature_array)) if feature_array[i] > 0]
    
    s.convergence=convergence_curve
    s.optimizer="SOA"   
    s.objfname=objf.__name__
    s.bestfitness = Leader_score
    s.bestf1 = Leader_f1
    s.bestIndividual = feature_index
    return s

X_train=np.load("D:\\Project minor\\minor_project_2\\numpy_files\\X_train.npy")
Y_train=np.load("D:\\Project minor\\minor_project_2\\numpy_files\\Y_train.npy")
X_test=np.load("D:\\Project minor\\minor_project_2\\numpy_files\\X_test.npy")
Y_test=np.load("D:\\Project minor\\minor_project_2\\numpy_files\\Y_test.npy")

X_train.shape 

sol = SOA(X_train=X_train, y_train=Y_train, X_test=X_test, y_test=Y_test,Max_iter=50)

print(sol.bestIndividual)

indices=sol.bestIndividual

print(len(sol.bestIndividual))

X_train_SOA=X_train[:,indices]
X_test_SOA=X_test[:,indices]

X_train_SOA.shape

np.save("numpy_files/X_train_SOA.npy",X_train_SOA)
np.save("numpy_files/Y_train_SOA.npy",Y_train)
np.save("numpy_files/X_test_SOA.npy",X_test_SOA)
np.save("numpy_files/Y_test_SOA.npy",Y_test)


















