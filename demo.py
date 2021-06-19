# This code is meant to provide efficient frontier for an asset universe selected from
# JpMorgan Capital Market Assumption

import pandas as pd
import numpy as np

xl_file = pd.read_excel("Input configuration.xlsx",sheet_name=None)

sheets = xl_file.keys()

inputconfig=xl_file['Input Config']
listheader=list(inputconfig)
print(listheader)
print(inputconfig.loc[:,listheader[1]])