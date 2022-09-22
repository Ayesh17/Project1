import numpy as np
import decision_trees as dt, data_storage as ds

file_name = "cat_dog_data.csv"

data = np.genfromtxt(file_name, dtype=str, delimiter=',')

a_samples,a_labels = ds.build_nparray(data)
print(type(a_labels))
print(a_samples)
print(a_labels,"\n")

l_samples,l_labels = ds.build_list(data)
print(type(l_labels))
for row in l_samples:
  print(row)
print(l_labels,"\n")

d_samples, d_labels = ds.build_dict(data)
print(type(d_labels))
for index in range(len(d_samples)):
    print(d_samples[index])
print(d_labels,"\n")

max_depth = 3
DT = dt.DT_train_binary(a_samples,a_labels,max_depth)
test_acc = dt.DT_test_binary(a_samples,a_labels,DT)
print("DT:",test_acc)
print(dt.DT_make_prediction([[1,1,1,0,1]],DT))


file_name2 = "data_2.csv"
data_2 = np.genfromtxt(file_name2, dtype=str, delimiter=',')
b_samples,b_labels = ds.build_nparray(data_2)
max_depth = 3
DT = dt.DT_train_real(b_samples,b_labels,max_depth)
test_acc = dt.DT_test_real(b_samples,b_labels,DT)
print("DT:",test_acc)

