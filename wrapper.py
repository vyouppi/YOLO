import numpy as np
import pandas as pd
from collections import OrderedDict


########## Load and Process user DATA Function #######


####### Load user input file data #######

def load_input(input_file):
    # Read input configuration user file
    user_input_config = pd.read_excel(input_file,sheet_name='Input Config',header=0)

    # Read capital market assumptions data
    cma_stat=pd.read_excel(input_file,sheet_name='Capital Market', header=0)
    correlation=pd.read_excel(input_file,sheet_name='Correlation',index_col=0, header=1)

    # store data in dictionary for ease of use later

    dict_data = {'cma_stat': cma_stat, 'correlation': correlation}

    return user_input_config, dict_data

# Select data filtered in line with the user asset list

def select_input_assets(user_input_config, dict_data):
    asset_list=user_input_config.loc[:,'Asset Class']
    cleaned_list=[x for x in asset_list if str(x) != 'nan']
    cma_stat=dict_data['cma_stat']
    correlation=dict_data['correlation']

    # Select from the global asset list only the assets within the user asset_list
    cma_stat=cma_stat.loc[asset_list,:]

    return 0


input_config, dict_input = load_input("Input configuration.xlsx")
#print(list(input_config))

allocation_data = input_config[input_config.index.isna()==False]
asset_dict=OrderedDict(zip(allocation_data.index,allocation_data['Asset Class']))
print(list(asset_dict.keys()))