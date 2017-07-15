# Script to load iris dataset from your local machine and perform simple operations.
# Load libraries
import pandas
data_location = "C:\Work\ML\iris.data" # Change this to match the location of the sample data on your local machine
data_features= ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
sample_dataset = pandas.read_csv(data_location, names=data_features)
# if data loads without issues, perform simple operations on dataset to get some facts
# Display sample rows
print("First 50 rows:\n", sample_dataset.head(50))
# Display data shape
print("Shape:\n", sample_dataset.shape)
# Display mean. median etc
print("Statistics:\n",sample_dataset.describe())
# Display class distribution
print("Distribution;\n", sample_dataset.groupby('class').size())