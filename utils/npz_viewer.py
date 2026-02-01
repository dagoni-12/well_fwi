from numpy import load
import os
print(os.path.abspath("."))
#data = load('~/diploma/utils/7_1-1_synthetic.npz')
data = load('data/synthetic/7_1-1_synthetic.npz')
lst = data.files
for item in lst:
    print(item)
    #print(data[item])