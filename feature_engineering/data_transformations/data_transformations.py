import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
#import codecademylib3_seaborn #this is a codecademy import - cant use it in local env
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#import file
coffee = pd.read_csv("starbucks_customers.csv")

#examine the features by print the columns
print(coffee.columns)

#look at each feature
print(coffee.info())

#set age feature to ages variable

ages = coffee['age']

# find min age
min_age = np.min(ages)

# find max age
max_age = np.max(ages)

#print the difference between min_age and max_age

print(max_age - min_age)

# get mean age and print 
mean_age = np.mean(ages)

#center the ages feature
centered_ages = ages - mean_age

#print centered ages
print(centered_ages)

#plot the data

# plt.hist(centered_ages)
# plt.show()

# find std dev of ages

std_dev_age = np.std(ages)

#standardize ages

ages_standardized = ((ages - mean_age) / std_dev_age)

#print mean and std dev

print(np.mean(ages_standardized))
print(np.std(ages_standardized))

#instantiate scaler
scaler = StandardScaler()

#reshape array
ages_reshaped = np.array(ages).reshape(-1, 1)

#standardize ages
ages_scaled = scaler.fit_transform(ages_reshaped)

#print mean and std dev

print(np.mean(ages_scaled))
print(np.std(ages_scaled))

#get the spent feature
spent = coffee['spent']

#find max spent
max_spent = np.max(spent)

#find min spent
min_spent = np.min(spent)

# find spent range
spent_range = max_spent - min_spent

#normalise spent column
spent_normalized = (spent - min_spent) / (max_spent - min_spent)

#print normalized spent array
print(spent_normalized)

# reshape array

spent_reshaped = np.array(spent).reshape(-1, 1)

# instatiate min max scaler
mmscaler = MinMaxScaler()

#normalize spent array

reshaped_scaled = mmscaler.fit_transform(spent_reshaped)

#print min and max from the array

print(np.min(reshaped_scaled))
print(np.max(reshaped_scaled))

#print ages min and max values
print(np.max(ages))
print(np.min(ages))

#define bin boundries (cut excludes endpoint (71 instead of 70))
age_bins = [12, 20, 30, 40, 71]

#create the binned column

coffee['binned_ages'] = pd.cut(coffee['age'], age_bins, right=False)

#print first 10 rows of binned ages

print(coffee['binned_ages'].head(10))

# plot the graph

# coffee['binned_ages'].value_counts().plot(kind='bar')
# plt.show()

#import cars

cars = pd.read_csv("cars.csv")

# get the sellingprice feature into prices variable

prices = cars['sellingprice']

# plot prices as a histogram, with 150 bins

plt.hist(prices, bins=150)
plt.show()

# log transform the prices feature

log_prices = np.log(prices)

# plot the log transformed prices as a hsitogram with 150 bins
plt.hist(log_prices, bins=150)
plt.show()