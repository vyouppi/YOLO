import math

import numpy as np
import pandas as pd
from collections import OrderedDict


########## Load and Process user DATA Function #######


####### Load user input file data #######

def load_input(input_file):
    # Read input configuration user file
    user_input_config = pd.read_excel(input_file,sheet_name='Input Config',header=0)

    # Read capital market assumptions data
    cma_stat=pd.read_excel(input_file,sheet_name='User_CMA', header=0)
    correlation=pd.read_excel(input_file,sheet_name='User_Correlation',header=0)

    # store data in dictionary for ease of use later

    dict_data = {'cma_stat': cma_stat, 'correlation': correlation}

    return user_input_config, dict_data

# Select data filtered in line with the user asset list

def select_input_assets(inputs, dict_data):
    asset_list=inputs['asset_list']
    cma_stat=dict_data['cma_stat']
    correlation=dict_data['correlation']

    # Select from the global asset list only the assets within the user asset_list

    arith_ret=cma_stat['Arith_Ret']
    geo_ret=cma_stat['Geo_Ret']
    vol=cma_stat['Vol']

    # computing covariance matrix

    temp_corr=correlation.to_numpy()
    temp_corr=temp_corr[:,1:]
    cov_mat=pd.DataFrame(np.diag(vol).T.dot(temp_corr).dot(np.diag(vol)))
    # aggregating the needed output

    outputs = {}

    outputs['arith_ret']=arith_ret
    outputs['geom_ret'] = geo_ret
    outputs['vol'] = vol
    outputs['correlation']=correlation
    outputs['cov']=cov_mat

    return outputs


input_config, dict_input = load_input("Input configuration.xlsx")
#print(list(input_config))

#allocation_data = input_config[input_config['Title']!='none']
#asset_dict=OrderedDict(zip(allocation_data.index,allocation_data['Asset Class']))
#print(input_config[input_config['Title']!='none'])
#print(allocation_data)

def collect_inputs(input_config, dict_data):

    alloc_data = input_config[input_config['Title'] != 'none']
    asset_dict = OrderedDict(zip(alloc_data.index,alloc_data['Title']))

    # get a list of all the 'proper' names

    asset_list = list(asset_dict.keys())
    columns = input_config.columns.tolist()

    # Collect alloc and params
    initial_allocation = np.array(alloc_data['Initial Allocation'])
    benchmark = np.array(alloc_data['Liability'])
    allocation_minima = np.array(alloc_data['Min Allocation'])
    allocation_maxima = np.array(alloc_data['Max Allocation'])


    active = {}

    active['annual cost'] =-np.array(alloc_data['fees'])
    active['alpha'] = np.array(alloc_data['alpha'])

    # collect group portfolio constraints
    group_list=columns[columns.index('Group Constraint1'):columns.index('Group Last')]
    group_name = input_config[group_list][input_config['Group Name']=='name'].iloc[0,:]
    group_constraints=input_config[group_list][input_config['Group Name']=='select'].iloc[0,:]
    group_constraints_min=input_config[group_list][input_config['Group Name']=='min'].iloc[0,:].astype('float64')
    group_constraints_max=input_config[group_list][input_config['Group Name']=='max'].iloc[0,:].astype('float64')
    #print(alloc_data[group_list])
    group_constraints_coeff = alloc_data[group_list].astype('float64')

    # Add  inputs
    inputs ={ 'asset_list':asset_list,
              'asset_dictionary':asset_dict,
              'initial_allocation': initial_allocation,
              'active':active,
              'benchmark':benchmark
              }


    # Amend Optimisation inputs
    inputs.update({'allocation_min':allocation_minima,
                   'allocation_max':allocation_maxima,
                   'group_name':group_name,
                   'group_list':group_list,
                   'group_constraints':group_constraints,
                   'group_constraints_min':group_constraints_min,
                   'group_constraints_max':group_constraints_max,
                   'group_constraints_coeff':group_constraints_coeff,
                   })

    # Align and filter data with covariance matrix

    inputstats = select_input_assets(inputs, dict_data)

    inputs.update(inputstats)


    return inputs

def convert_arith_to_geo(arith,vol):

    return arith-0.5*vol*vol


def initial_stats(input_config,dict_input):

    # compute the volatility defined as sqrt (Vol.T.dot(cov).vol)

    inputs = collect_inputs(input_config, dict_input)
    vol = inputs['vol']
    cov = inputs['cov'].to_numpy()
    arith= inputs['arith_ret']
    init_alloc = inputs['initial_allocation'] # array numpy

    ptf_vol=math.sqrt(init_alloc.T.dot(cov).dot(init_alloc))
    ptf_ret_Arith=init_alloc.dot(arith)
    ptf_ret_geo = convert_arith_to_geo(ptf_ret_Arith,ptf_vol)

    print(ptf_vol)
    print(ptf_ret_Arith)
    print(ptf_ret_geo)

#a=collect_inputs(input_config,dict_input)
b = initial_stats(input_config,dict_input)

