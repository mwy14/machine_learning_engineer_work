import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
#import codecademylib3_seaborn #this is a codecademy import - cant use it in local env

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

plt.hist(centered_ages)
plt.show()
