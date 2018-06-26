"""

Author   : Manu Aatitya R P
Language : Python 3

Multivariate Linear regression to predict FIFA Rankings
based on data for the past 25 years

Submitted for Delta Inductions 2k18 Open Profile ML Normal Mode 

Normal Equation method used instead of Gradient Descent
A*X = Y 
A = input train examples with bias
X = parameters matrix 
Y = output for the train examples

No feature scaling required  Parameters computed analytically

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

# To predict ranking       
def predict_rankings(Test,theta,Y_test):
    Y = np.matmul(Test,theta)
    correct_answer = 0
    for i in range (len(Y) - 1):
        if (Y[i] - Y_test[i]) < 0.12:
            correct_answer +=1
    accuracy = (correct_answer/(len(Y))) * 100
    return accuracy

# function to estimate coefficient
def estimate_coefficient(A,y):
    theta = np.matmul(np.matmul(np.linalg.pinv(np.matmul(np.transpose(A),A)),np.transpose(A)),y)
    # theta = (((A)'*A)^-1)*(A')*y    ' = transpose
    return theta

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
    parameters = ['rank_date','previous_points','total_points']
    predicted_results = []
    for i in worldcup_teams:
        country_data = data[data.country_full == i]
        A = np.ones(3*len(country_data)//4)
        Test = np.ones(len(country_data) - (3*len(country_data)//4))
        Y_train = np.array([list(country_data['rank'])[0:(3*len(country_data['rank']))//4]])
        Y_test = np.array([list(country_data['rank'])[(3*len(country_data['rank']))//4 + 1 :]])
        for j in parameters :
            A = np.vstack([A,np.array(list(country_data[j]))[0:(3*len(country_data[j]))//4]])
            Test = np.vstack([Test,np.array(list(country_data[j]))[(3*len(country_data[j]))//4 :len(country_data[j]) + 1]])
        A = A.transpose()
        Y_train = Y_train.transpose()
        Y_test = Y_test.transpose()
        Test = Test.transpose()
        # computing coefficients for parameters
        theta = estimate_coefficient(A,Y_train)
        predicted_result = predict_rankings(Test,theta,Y_test)
        # To predict accuracy
        predicted_results.append([i,predicted_result])
    # Printing output
    print('\t\t  Country_Name','\t','Accuracy')  
    for i in range(len(predicted_results)):
      print('{0}'.format(predicted_results[i][0]).rjust(30),'\t','{0:.2f}'.format(predicted_results[i][1]))  

# Main function call
if __name__ == '__main__' :
    main()
