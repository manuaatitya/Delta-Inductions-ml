# import necessary modules
import numpy as np
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    data = pd.read_csv('Wine.csv')
    return data

def distance(a,b,ax = 1):
    return np.linalg.norm(a-b,axis = ax)
    
def main():
  data = load_data()
  alcohol_content = np.array(data[data.columns[1]],dtype = float)
  ash_content = np.array(data[data.columns[3]],dtype = float)
  hue_content = np.array(data[data.columns[11]],dtype = float)
  malic_acid = np.array(data[data.columns[2]],dtype = float)
  magnesium_content = np.array(data[data.columns[5]],dtype = float)
  total_phenols = np.array(data[data.columns[6]],dtype = float)
  
  X = np.array(list(zip(alcohol_content, ash_content,hue_content,malic_acid,magnesium_content,
                        total_phenols)))
  no_clusters = 6
  c_x = np.random.randint(0,np.max(X) - 15,size = no_clusters)
  c_y = np.random.randint(0,np.max(X) - 15 , size = no_clusters)
  c_z = np.random.randint(0,np.max(X) - 15 , size = no_clusters)
  c_a = np.random.randint(0,np.max(X) - 15 , size = no_clusters)
  c_b = np.random.randint(0,np.max(X) - 15 , size = no_clusters)
  c_d = np.random.randint(0,np.max(X) - 15 , size = no_clusters)
  centroids = np.array(list(zip(c_x,c_y,c_z,c_a,c_b,c_d)),dtype = float)
  
  c_old = np.zeros(centroids.shape)
  cluster_labels = np.zeros(len(X))
  error = distance(centroids,c_old,None)
  while error != 0:
      
      for i in range(len(X)):
          
          dist = distance(X[i],centroids)
          cluster = np.argmin(dist)
          cluster_labels[i] = cluster
          c_old = deepcopy(centroids)
          
          
          for j in range(no_clusters):
              
              points = [X[k] for k in range(len(X)) if cluster_labels[k] ==  j]
              
              centroids[j] = np.mean(points,axis = 0)
              
          error = distance(centroids,c_old,None)
          
  colors = ['red','yellow','blue','black','green','violet']
  for i in range(no_clusters):
      
      points = np.array([X[j] for j in range(len(X)) if cluster_labels[j] == i])
      plt.scatter(points[:, 0], points[:, 1], s=7, c = colors[i])
      
  plt.title('K means clustering\n wine samples separated into 6 groups \n projection pf 6d points into 2d plane \n for visualizing clustering') 
  plt.xlabel('Alcohol_content')
  plt.ylabel('Ash_content')
  plt.show()
  
if __name__ == "__main__":
  main()