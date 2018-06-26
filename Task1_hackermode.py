"""

Author   : Manu Aatitya R P
Language : Python 3

Multivariate Linear regression to predict FIFA Rankings
based on data for the past 25 years

Submitted for Delta Inductions 2k18 Open Profile ML Hacker Mode

Gradient descent used with LASSO regularisation

75 % input used to train the model and remaining used to test the same

Results in format %country_name %accuracy
 
"""
# import necessary modules
import datetime as dt
import pandas as pd
import numpy as np

# function to return names of worldcup teams
def get_worldcup_teams(file):
    data = []
    for i in file['country_full']:
        if i not in data:
            data.append(i)
    return data

        
def predict_rankings(x,y,theta,z,w,v,u):
   x = np.array(x)
   z = np.array(z)
   w = np.array(w)
   m = np.size(x)
   y = np.array(y)
   v = np.array(v)
   u = np.array(u)
   predicted_ranking = (theta[0] + theta[1] * x + theta[2] * z + theta[3] * w + theta[4] * v + theta[5] * u) # * (x_max - x_min) + x_mean
   correct_answer = 0
   for i in range (m):
       if (predicted_ranking[i] - y[i])  < 0.1:
           correct_answer +=1
   accuracy = (correct_answer/m) * 100
   return accuracy

# function to estimate coefficient
def estimate_coefficient(x,y,z,w,v,u):
    # plt.title('Country ranking over the years')
    # plt.xlabel('Iterations')
    # plt.ylabel('Cost function')
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    w = np.array(w)
    v = np.array(v)
    u = np.array(u)
    size = np.size(x)
    iterations = 1000
    # iter_array = [i+1 for i in range(1000)]
    theta = [0,0,0,0,0,0]
    temp = [0,0,0,0,0,0]
    alpha = 0.01   # verified by plugging in values
    Lambda = 1   # Regularization parameter
    # J_theta =[]
    for i in range(iterations):
        # J_theta.append(np.sum(((theta[0] + theta[1] * x + theta[2] * z + theta[3] * w ) - y)**2)/(2*size))
        h_theta = theta[0] + theta[1] * x + theta[2] * z + theta[3] * w + theta[4] * v + theta[5] * u
        temp[0] = theta[0] - ((alpha) * (np.sum((h_theta - y))/size))
        temp[1] = theta[1] - alpha * (np.sum((h_theta - y)*x)/size - Lambda * (theta[1]/size))
        temp[2] = theta[2] - alpha * (np.sum((h_theta - y)*z)/size - Lambda * (theta[2]/size))
        temp[3] = theta[3] - alpha * (np.sum((h_theta - y)*w)/size - Lambda * (theta[3]/size))
        temp[4] = theta[4] - alpha * (np.sum((h_theta - y)*w)/size - Lambda * (theta[4]/size))
        temp[5] = theta[5] - alpha * (np.sum((h_theta - y)*w)/size - Lambda * (theta[5]/size))
        theta = temp
    # plt.plot(iter_array,J_theta,color = 'g')
    # plt.show()
    return theta

# function to scale the features
def feature_scaling(x):
    if (np.max(x) != np.min(x)):
        x = (x - np.mean(x)) / (np.max(x) - np.min(x))
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
        # Final output compare
        country_ranking = feature_scaling(list(country_data['rank']))
        country_ranking_train = country_ranking[0:(3*len(country_ranking))//4]
        country_ranking_test = country_ranking[(3*len(country_ranking))//4+1:]
        # First feature date of ranking
        country_year = feature_scaling(list(country_data['rank_date']))
        country_year_train = country_year[0:(3*len(country_year))//4]
        country_year_test = country_year[(3*len(country_year))//4+1:]
        # Second feature previous points
        country_previous_points = feature_scaling(list(country_data['previous_points']))
        country_previous_points_train = country_previous_points[0:(3*len(country_previous_points))//4]
        country_previous_points_test = country_previous_points[(3*len(country_previous_points))//4+1:]
        # Third feature total points
        country_total_points = feature_scaling(list(country_data['total_points']))
        country_total_points_train = country_total_points[0:(3*len(country_total_points))//4]
        country_total_points_test = country_total_points[(3*len(country_total_points))//4+1:]
        # Fourth feature rank change
        country_rank_change = feature_scaling(list(country_data['rank_change']))
        country_rank_change_train = country_rank_change[0:(3*len(country_rank_change))//4]
        country_rank_change_test = country_rank_change[(3*len(country_rank_change))//4+1:]
        # Fifth feature curr year average
        country_cur_year_avg = feature_scaling(list(country_data['cur_year_avg']))
        country_cur_year_avg_train = country_cur_year_avg[0:(3*len(country_cur_year_avg))//4]
        country_cur_year_avg_test = country_cur_year_avg[(3*len(country_cur_year_avg))//4 + 1 :]
        # Estimating gradient descent parameters
        theta = estimate_coefficient(country_year_train,country_ranking_train,country_total_points_train,country_previous_points_train,country_rank_change_train,country_cur_year_avg_train)
        predicted_result = predict_rankings(country_year_test,country_ranking_test,theta,country_total_points_test,country_previous_points_test,country_rank_change_test,country_cur_year_avg_test)
        predicted_results.append([i,predicted_result])
    # Printinh output
    print('\t\tCountry_Name','\t','Accuracy')  
    for i in range(len(predicted_results)):
      print('{0}'.format(predicted_results[i][0]).rjust(30),'\t','{0:.2f}'.format(predicted_results[i][1]))  
    

# Main function call
if __name__ == '__main__' :
    main()