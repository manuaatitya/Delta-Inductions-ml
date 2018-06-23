"""

Author   : Manu Aatitya R P
Language : Python 3

Multivariate Linear regression to predict FIFA Rankings
based on data for the past 25 years

Submitted for Delta Inductions 2k18 Open Profile ML 
Task 1 Normal Mode

Variables used : Year of Rankings, Previous Points, Total Points

Feature Scaling used in the dataset
 
"""
# import necessary modules
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# function to return names of worldcup teams
def get_worldcup_teams(file):
    data = []
    for i in file['country_full']:
        if i not in data:
            data.append(i)
    return data

        
def predict_rankings(country_year_test,country_ranking_test,theta,country_total_points_test,country_previous_points_test):
   x = np.array(country_year_test)
   z = np.array(country_total_points_test)
   w = np.array(country_previous_points_test)
   m = np.size(x)
   # x_mean = np.mean(x)
   # x_max = np.max(x)
   # x_min = np.min(x)
   y = np.array(country_ranking_test)
   # y = y * (np.max(y) - np.min(y)) + np.mean(y)
   predicted_ranking = (theta[0] + theta[1] * x + theta[2] * z + theta[3] * w) # * (x_max - x_min) + x_mean
   correct_answer = 0
   for i in range (m):
       if (predicted_ranking[i] - y[i])  < 0.18:
           correct_answer +=1
   accuracy = (correct_answer/m) * 100
   return accuracy

# function to estimate coefficient
def estimate_coefficient(x,y,z,w):
    # plt.title('Country ranking over the years')
    # plt.xlabel('Iterations')
    # plt.ylabel('Cost function')
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    w = np.array(w)
    size = np.size(x)
    iterations = 1000
    # iter_array = [i+1 for i in range(1000)]
    theta = [0,0,0,0]
    temp = [0,0,0,0]
    alpha = 0.1   # verified by plugging in values
    # J_theta =[]
    for i in range(iterations):
        # J_theta.append(np.sum(((theta[0] + theta[1] * x + theta[2] * z + theta[3] * w ) - y)**2)/(2*size))
        h_theta = theta[0] + theta[1] * x + theta[2] * z + theta[3] *w
        temp[0] = theta[0] - ((alpha) * (np.sum((h_theta - y))/size))
        temp[1] = theta[1] - ((alpha) * (np.sum((h_theta - y)*x)/size)) 
        temp[2] = theta[2] - ((alpha) * (np.sum((h_theta - y)*z)/size))
        temp[3] = theta[3] - ((alpha) * (np.sum((h_theta - y)*w)/size))
        theta = temp
    # plt.plot(iter_array,J_theta,color = 'g')
    # plt.show()
    return theta

# function to scale the features
def feature_scaling(x):
    x_mean = np.mean(x)
    x_max = np.max(x)
    x_min = np.min(x)
    x = (x - x_mean) / (x_max - x_min)
    return x

# function to import data
def collect_data():
    data = pd.read_csv('fifa_ranking.csv') 
    return data
       
# Main driver function
def main():
    data = collect_data()
    np.seterr(all='warn')
    # to convert date to ordinal format
    data['rank_date'] = pd.to_datetime(data['rank_date'])
    data['rank_date'] = data['rank_date'].map(dt.datetime.toordinal)
    worldcup_teams = get_worldcup_teams(data)
    predicted_results = []
    for i in worldcup_teams:
        country_data = data[data.country_full == i]
        country_ranking = list(country_data['rank'])
        country_ranking = feature_scaling(country_ranking)
        country_year = list(country_data['rank_date'])
        country_year = feature_scaling(country_year)
        country_previous_points = list(country_data['previous_points'])
        country_previous_points = feature_scaling(country_previous_points)
        country_total_points = list(country_data['total_points'])
        country_total_points = feature_scaling(country_total_points)
        country_year_train = country_year[0:(3*len(country_year))//4]
        country_year_test = country_year[(3*len(country_year))//4+1:]
        country_previous_points_train = country_year[0:(3*len(country_year))//4]
        country_previous_points_test = country_year[(3*len(country_year))//4+1:]
        country_total_points_train = country_year[0:(3*len(country_year))//4]
        country_total_points_test = country_year[(3*len(country_year))//4+1:]
        country_ranking_train = country_ranking[0:(3*len(country_ranking))//4]
        country_ranking_test = country_ranking[(3*len(country_ranking))//4+1:]
        theta = estimate_coefficient(country_year_train,country_ranking_train,country_total_points_train,country_previous_points_train)
        predicted_result = predict_rankings(country_year_test,country_ranking_test,theta,country_total_points_test,country_previous_points_test)
        predicted_results.append([i,predicted_result])
    
    print('\t\tCountry_Name','\t','Accuracy')  
    for i in range(len(predicted_results)):
      print('{0}'.format(predicted_results[i][0]).rjust(30),'\t','{0:.2f}'.format(predicted_results[i][1]))  
    # plot_regression_line(india_year,india_rankings,theta)

# Main function call
if __name__ == '__main__' :
    main()