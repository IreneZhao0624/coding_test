#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 14:35:27 2019

@author: Irene
"""
import os 
import re
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from mord import LogisticAT
import statsmodels.api as sm 
from sec_edgar_downloader import Downloader
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import make_scorer
# from SECEdgar.filings import Filing


DATE = []
Name = []
CUSIP = []
Market_value = []
Share = []

def num(i):
    re1='.*?'	# Non-greedy match on filler
    re2='(\\d+)'	# Integer Number 1
    rg = re.compile(re1+re2,re.IGNORECASE|re.DOTALL)
    m = rg.search(i)
    int1=m.group(1)
    return int1

def issuer(i):
    re1='.*?'	# Non-greedy match on filler
    re2='(?:[a-z][a-z]+)'	# Uninteresting: word
    re3='.*?'	# Non-greedy match on filler
    re4='((?:[a-z][a-z]+))'	# Word 1
    rg = re.compile(re1+re2+re3+re4,re.IGNORECASE|re.DOTALL)
    m = rg.search(i)
    word=m.group(1)
    return word

def cusip(i):
    re1='.*?'	# Non-greedy match on filler
    re2='(\\d+)'	# Integer Number 1
    re3='((?:[a-z][a-z0-9_]*))'	# Variable Name 1
    rg = re.compile(re1+re2+re3,re.IGNORECASE|re.DOTALL)
    m = rg.search(i)
    if m != None:
        int1=m.group(1)
        var1=m.group(2)
        string = int1+var1
        return string
    else:
        re1='.*?'	# Non-greedy match on filler
        re2='(\\d+)'	# Integer Number 1

        rg = re.compile(re1+re2,re.IGNORECASE|re.DOTALL)
        m = rg.search(i)
        int1=m.group(1)
        return int1

def reader(file):
    """
        Finds the following strings in a document
    """
    f = open(file, 'r')
    global placeholder
    count = 0
    for i in f:
        if 'CONFORMED PERIOD OF REPORT' in i:
            int1 = num(i)
            placeholder = int1
        if '<nameOfIssuer>' in i:
            word = issuer(i)
            Name.append(word)
        if '<cusip>' in i:
            string = cusip(i)
            CUSIP.append(string)
        if '<value>' in i:
            int1 = num(i)
            Market_value.append(int1)
        if '<sshPrnamt>' in i:
            int1 = num(i)
            Share.append(int1)
            count += 1
    DATE.extend([placeholder] * count)


def save(file, name):
    """
        Save reports to csv files
    """
    reader(file)
    raw_data = {'DATE': DATE, 'Name': Name,
                'CUSIP':CUSIP, 'Market_value':Market_value,
                'Share':Share}
    df = pd.DataFrame(raw_data, columns = ['DATE', 'Name', 'CUSIP', 'Market_value', 'Share'])
    df.to_csv(name+'.csv')
    return df

path = '/Users/Irene/Desktop/M/sec_edgar_filings/'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.txt' in file:
            files.append(os.path.join(r, file))
for f in files:
    if '1061165' in f:
        LPC = save(f, 'LPC')
    if '1103804' in f:
        VG = save(f, 'VG')
    if '1061768' in f:
        BG = save(f, 'BG')
        
for fund in ['LPC','VG','BG']:
    fund = pd.read_csv(path + fund)
    
# we keep LPC and BG for simplicity

LPC = pd.read_csv('/Users/Irene/Desktop/M/'+'1061165.csv')
LPC.head()
LPC = LPC.drop(['CIK','CUSIP','amend_x', 'period_focus_x', 'fiscal_year_x','doc_type_x'],axis = 1)
LPC = pd.DataFrame(LPC.iloc[:,1:])
LPC = LPC.sort_values(by = ['symbol','end_date'])

# remove those symbols with counts less than 10 (we have 25 data points in total)
# 201306 - 201903
LPC = LPC.groupby('symbol').filter(lambda x : len(x)>= 10 and len(x) <= 24).drop_duplicates()
LPC.symbol.value_counts().plot('bar')
LPC.to_csv('/Users/Irene/Desktop/M/'+'LPC_clean.csv')

LPC['Price'] = LPC['Market value']*1000/LPC['Share']
LPC = LPC.dropna(axis = 0,how='any').drop_duplicates()
LPC = LPC.groupby('symbol').filter(lambda x : len(x)<= 24)
LPC.symbol.value_counts().plot('bar')

LPC['Change in share'] = LPC.groupby('symbol')['Share'].pct_change()
LPC.columns
LPC.to_csv('/Users/Irene/Desktop/M/'+'BG_clean.csv')

#--------- BG fund -----------------#
BG = pd.read_csv('/Users/Irene/Desktop/M/'+'1061768.csv')
BG.head()
BG = BG.drop(['CIK','CUSIP','amend_x', 'period_focus_x', 'fiscal_year_x','doc_type_x'],axis = 1)
BG = pd.DataFrame(BG.iloc[:,1:])
# calculate implied stock price
BG['Price'] = BG['Market value']*1000/BG['Share']
# change of position

BG = BG.sort_values(by = ['symbol','end_date'])

# remove those symbols with counts less than 10 (we have 25 data points in total)
# 201306 - 201903
#BG = BG.groupby('symbol').filter(lambda x : len(x)>= 10 and len(x) <= 24).drop_duplicates()

'''
#------------ NG holding---------------#
h_NG = BG.loc[BG.symbol == 'NG'].dropna(axis = 0,thresh = 8)
h_NG = h_NG.replace(0,None).dropna(axis=1,thresh = 15).drop('dividend_x',axis = 1)
h_NG['Change in share'] = h_NG['Share'].pct_change()
'''

#----------- Linear regression to analyze holding percentage change --------#

BG = BG.dropna(axis = 0,how='any').drop_duplicates()
BG = BG.groupby('symbol').filter(lambda x : len(x)<= 24)
BG.symbol.value_counts().plot('bar')

BG['Change in share'] = BG.groupby('symbol')['Share'].pct_change()
BG.columns
BG.to_csv('/Users/Irene/Desktop/M/'+'BG_clean.csv')

# normalise by tickers
# load data
BG = pd.read_csv('/Users/Irene/Desktop/M/BG_clean.csv')

# select columns for normalization 
names = BG.iloc[:,5]
data = BG.iloc[:,6:-2]

def get_positions():
    """
        Get positions of each stock group
    """
    positions = ['0']
    init_counter = 0
    counter = 0 
    for i in range(len(names) - 1):
        if names[i + 1] == names[i]:
            counter += 1
        else:
            counter += 1
            positions.append(counter)
    positions.append(len(names))
    return positions

def create_index_names(start, finish):
    """
        Create dataframe index numbers
    """
    index_names = []
    for i in range(start, finish):
        index_names.append(str(i))
    return index_names

def normalize_data():
    """
        Firstly in the for loop we get a stock group from dataframe,
        then we normalize the batch and append it to a list 
    """
    normalized_data = []
    positions = get_positions()

    for i in range(len(positions) -1):
        difference = int(positions[i+1]) - int(positions[i])
        start = int(positions[i+1])- difference
        finish = positions[i+1]
        scaler = Normalizer().fit(data[start:finish])
        normalizedX = scaler.transform(data[start:finish])
        
        temp_normalized_data = pd.DataFrame(normalizedX, index = create_index_names(start,finish),
            columns=['revenues_x', 'op_income_x', 'net_income_x',
                'eps_basic_x', 'eps_diluted_x', 'dividend_x	', 'assets_x',
                'cur_assets_x', 'cur_liab_x	', 'cash_x', 'equity_x',
                'cash_flow_op_x', 'cash_flow_inv_x', 'cash_flow_fin_x'], dtype='float')
        normalized_data.append(temp_normalized_data)

    return normalized_data

# convert data into dataframe
raw_data = normalize_data()
normalized_data = pd.concat(raw_data)

# copy data 
normalized_df = BG.copy().drop(columns=BG.iloc[:,6:-2])

# save to csv 
normalized_data.to_csv('normalized_data.csv')
normalized_df.to_csv('normalized_df.csv')

# load csv
files = ['normalized_df.csv', 'normalized_data.csv']
dfs = [pd.read_csv(f, sep=",") for f in files]

# final data
final_cleaned_data = pd.concat(dfs, axis=1).iloc[:,1:]

#BG.iloc[:,4:(len(BG.columns)-1)].groupby('symbol').transform(lambda x: (x - x.mean()) / x.std())

BG.head()

# linear regression
def Linear_Regression(df):
    # split dataset into predictors and dependent variable
    X = df[regressors].shift(1).dropna() # use lag 1
    y = df['Change in share'].iloc[1:,]
    
    # X = pd.get_dummies(data=X, drop_first=True) # convert categorical variable into dummies 
                                                # and drop one level for each
    X = sm.add_constant(X)                      # allows for an intercept
    
    model = sm.OLS(y, X).fit()
    
    return model

BG.columns

regressors = ['Price', 'op_income_x',
       'net_income_x', 'eps_basic_x', 'eps_diluted_x', 'assets_x',
       'cur_assets_x', 'cur_liab_x\t', 'cash_x', 'equity_x', 'cash_flow_inv_x',
       'dividend_x\t']

BG = final_cleaned_data
BG.columns
model = Linear_Regression(BG)
model.summary()

feature = model.pvalues.to_frame()
feature.columns = ['p-value']
feature['score'] = 1/(feature['p-value']+0.1)
feature = feature.sort_values(by = 'score', ascending = False )
# feature cut-offs
# feature_imp = feature[:10]
#feature_imp[feature_imp['p-value'] <= 0.1].plot.bar(figsize = (14,12))

plt.figure(figsize=(20,10))
plt.barh(feature.index,feature['score'],color='mediumblue')
plt.title('Score for each feature')
plt.ylabel('Feature')
plt.xlabel('Score of importance')
plt.show()

#conclusion: we find there are few features of the listed companies 
# can explain the percentange change in the holdings for BG fund. 
# They are ranked by the importance (based on the p-values):
# cash flows (-), stock price (+), dividend (+), equity (-), current asset (+)

# Further analysis can be done by finding more macro indicators, sector indicators
# We can also try how does those fundamentals along with market indicator influence 
# the allocation of sectors in stead of a specific stock

#----------- Logistic regression to analyze holding increase/decrease--------#
BG = pd.read_csv('/Users/Irene/Desktop/M/'+'1061768.csv')
BG.head()
BG = BG.drop(['CIK','CUSIP','amend_x', 'period_focus_x', 'fiscal_year_x','doc_type_x'],axis = 1)
BG = pd.DataFrame(BG.iloc[:,1:])
# calculate implied stock price
BG['Price'] = BG['Market value']*1000/BG['Share']
# change of position

BG = BG.sort_values(by = ['symbol','end_date'])
h_NG = BG.loc[BG.symbol == 'NG'].dropna(axis = 0,thresh = 7)
h_NG = h_NG.replace(0,None).dropna(axis=1,thresh = 15).drop('dividend_x',axis = 1)
h_NG['Change in share'] = h_NG['Share'].pct_change()

def direction(df):
    if df['Change in share'] > 0:
        return 1
    elif df['Change in share'] == 0:
        return 0 
    else:
        return -1 


h_NG['Change direction'] = h_NG.apply(direction, axis = 1)

# instantiate models
model_linear = LinearRegression()
model_multi = LogisticRegression(multi_class='multinomial',
    solver='lbfgs',
    class_weight='balanced')
model_ordinal = LogisticAT(alpha=0)  # alpha parameter set to zero to perform no regularisation

# divide df into features matrix and target vector
features = h_NG[regressors].shift(1).dropna()
target = h_NG['Change direction'][1:]

MAE = make_scorer(mean_absolute_error)
folds = 5

print('Mean absolute error:' )
MAE_linear = cross_val_score(model_linear,
    features,
    target,
    cv=folds,
    scoring=MAE)
print('Linear regression: ', np.mean(MAE_linear))
MAE_1vR = cross_val_score(model_1vR,
    features,
    target,
    cv=folds,
    scoring=MAE)

print('Logistic regression (multinomial): ', np.mean(MAE_multi))
MAE_ordinal = cross_val_score(model_ordinal,
    features,
    target,
    cv=folds,
    scoring=MAE)
print('Ordered logistic regression: ', np.mean(MAE_ordinal))

# Conclusion: multinomial model is the best amongst those 3 models, 
# linear regression, multinomial logistic regression, ordered logistic regression,
# in predicting the direction of change on holdings: increase, unchanged, derease.

# For further analysis, we can also try neural networks to model the changes 
# in the holding direction if the data can trace back to 1990s. 
