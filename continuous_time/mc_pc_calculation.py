import time
from multiprocessing import Pool
import pickle

from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
import pandas as pd

#@title Functions for training readout layer

def generate_offsets(input, reservoir_state, tau):
  if tau == 0:
    return input, reservoir_state[1:]
  elif tau > 0:
    return input[tau:], reservoir_state[1:-tau]
  else:
    return input[:tau], reservoir_state[1-tau:]

def train_model(x, y, model):
    # Flatten reservoir states for use as input features
    emissions = np.array(y)  # Target outputs (emissions)

    split_index = int(0.8 * len(x))  # 80% for training, 20% for testing

    # Split into training and testing sets
    X_train, X_test = x[:split_index], x[split_index:]
    emissions_train, emissions_test = y[:split_index], y[split_index:]

    # Initialize and train the regressor
    model = model # RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, emissions_train)

    # EVALUATING ON TRAINING
    emissions_pred_train = model.predict(X_train)
    r2_train =  r2_score(emissions_train, emissions_pred_train)
    mse_train = mean_squared_error(emissions_train, emissions_pred_train)

    # EVALUATE ON TESTING
    emissions_pred_test = model.predict(X_test)
    r2_test = r2_score(emissions_test, emissions_pred_test)
    mse_test = mean_squared_error(emissions_test, emissions_pred_test)

    return model, r2_train, r2_test, mse_train, mse_test

def testing_training_metrics(r2_train, r2_test, mse_train, mse_test):
  data = {
      " ": ["R^2", "MSE"],
      "Training": [r2_train, mse_train],
      "Testing": [r2_test, mse_test]
  }

  df = pd.DataFrame(data)

  return df

reservoir_type = "noisy_limit"

def f(tau):
        with open('x_list.pkl', 'rb') as file:
            x_list = pickle.load(file)
        with open('h_x_'+reservoir_type+'.pkl', 'rb') as file:
            h_x = pickle.load(file)
        y, x = generate_offsets(x_list, h_x, tau = tau)
        # print("Offset:", tau)
        # print("lengths:", len(y), len(x))
        x = [[i] for i in x] # fix dimensions
        model = LinearRegression()
        model,r2_train, r2_test, mse_train, mse_test = train_model(x, y, model)
        return r2_train
    

if __name__ == "__main__":
    

    print("Calculating r^2 values")
    tau_range = range(-40000, 40000, 500)
    
    # without 

    # print("Start Time:", time.ctime())
    # start_time = time.time()
    # r2 = []
    # with open('x_list.pkl', 'rb') as file:
    #         x_list = pickle.load(file)
    # with open('h_x.pkl', 'rb') as file:
    #         h_x = pickle.load(file)
    # for i in tqdm(tau_range):
    #     y, x = generate_offsets(x_list, h_x, tau = i)
    #     # print("Offset:", tau)
    #     # print("lengths:", len(y), len(x))
    #     x = [[i] for i in x] # fix dimensions
    #     model = LinearRegression()
    #     model,r2_train, r2_test, mse_train, mse_test = train_model(x, y, model)
    #     r2.append(r2_train)
    # # print(r2)
    # print("Without pooling,", time.time() - start_time, "to run")
    # plt.plot(tau_range, r2, 'o')
    # plt.savefig("1.png")
    # plt.clf()
    
    start_time = time.time()
    print("Start Time:", time.ctime())
    with Pool() as pool:
        results = pool.map(f, tau_range)
    
    with open('lorenz_'+reservoir_type+'_mcpc.pkl', 'wb') as file:
        pickle.dump(results, file)

    plt.plot(tau_range, results, 'o', )
    plt.savefig("lorenz_"+reservoir_type+"_mcpc.png")
    print("With pooling,", time.time() - start_time, "seconds to run")
