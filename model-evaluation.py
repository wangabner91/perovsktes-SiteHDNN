import numpy as np
import torch
import math

#R2
def nihe(y_yuce, y_zhenzhi):
    y_zhenzhi_mean = torch.mean(y_zhenzhi)
    nihe_value = 1 - sum((y_yuce - y_zhenzhi) ** 2)/sum((y_zhenzhi - y_zhenzhi_mean) ** 2)
    return nihe_value

#MAE
def mae(y_yuce, y_zhenzhi):
    mae_value = torch.mean(abs(y_yuce - y_zhenzhi))
    return mae_value

#RMSE
def rmse(y_yuce, y_zhenzhi):
    rmse_value = math.sqrt(torch.mean((y_yuce - y_zhenzhi) ** 2))
    return rmse_value
    
#read file
def Elist_str2values(list_str):
    list_float = []    
    len_str = len(list_str)
    for i in range(1,len_str):
        list_word = list_str[i].split()
        word_nums = len(list_word)        
        list_line_float = []
        for j in range(word_nums):
            element_float = float(list_word[j])
            list_line_float.append(element_float)        
        list_float.append(list_line_float)
    return list_float

############################
###The "energy.out" file is the output file for model prediction. 
############################
with open("energy.out", 'r') as file:
    lines = file.readlines()
energy_list = []

for str in lines:
    energy_list.append(str)

energy_values = Elist_str2values(energy_list)
energy_value2 = torch.tensor(np.array(energy_values)).to(torch.float)
####################################
energy_ture = energy_value2[:,2]
energy_yuce = energy_value2[:,3]

fitting_value = nihe(energy_ture,energy_yuce)
print(f"fitting coefficient: {fitting_value}")

mae_value = mae(energy_ture,energy_yuce)
print(f"MAE: {mae_value}")

rmse_value = rmse(energy_ture,energy_yuce)
print(f"RMSE: {rmse_value}")



mae_value_atom = torch.mean(abs(energy_value2[:,4]))
print(f"MAE_atom: {mae_value_atom}")

rmse_value_atom = math.sqrt(torch.mean((energy_value2[:,4]) ** 2))
print(f"RMSE_atom: {rmse_value_atom}")