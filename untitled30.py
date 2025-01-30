#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 20:57:33 2025

@author: varuundeshpande
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests, adfuller


'''
Look at JPMC FFIEC-031 form for codes
'''

'''
#Total Uninsured Deposits are
#computed as the sum of total foreign deposits and domestic deposit accounts with balances over
#$100,000 before 2006Q1, non-retirement accounts with balances above $100,000 and retirement
#accounts with balances above $100,000 for the time period of 2006Q1-2009Q2
#and all deposits over $250,000 after 2009Q2 (reflecting the temporary increase in deposit insurance limit later
#made permanent), reported in schedule RC-O

#RCONF051 - Amount of deposit accounts (excluding retirement accounts) of more than $250,000
#RCONF047 - Amount of retirement deposit accounts of more than $250,000
#total unisured deposits = RCONF047 + RCONF051

#Uninsured Time Deposits are time deposits above $100,000 till 2010Q1 and above $250,000 after 2010Q1 plus foreign interest bearing deposits
#RCONJ474 (RCE-I) - Unisured time deposits
#RCFN3404 (RC-K) - Interest-bearing deposits in foreign offices, EDGE and Agreement subsidiaries, and IBFs

'''

base_path = '/Users/varuundeshpande/Desktop/Columbia_MSFE/Banking_paper_2025'

data_path = os.path.join(base_path, 'Data')

def data_read():
    '''
    Reads all the file names for data between 2010Q1 to 2024Q3

    Returns
    -------
    None.

    '''
    file_names = ['FFIEC CDR Call Bulk All Schedules 03312010',
                     'FFIEC CDR Call Bulk All Schedules 06302010',
                     'FFIEC CDR Call Bulk All Schedules 09302010',
                     'FFIEC CDR Call Bulk All Schedules 12312010',
                     'FFIEC CDR Call Bulk All Schedules 03312011',
                     'FFIEC CDR Call Bulk All Schedules 06302011',
                     'FFIEC CDR Call Bulk All Schedules 09302011',
                     'FFIEC CDR Call Bulk All Schedules 12312011',
                     'FFIEC CDR Call Bulk All Schedules 03312012',
                     'FFIEC CDR Call Bulk All Schedules 06302012',
                     'FFIEC CDR Call Bulk All Schedules 09302012',
                     'FFIEC CDR Call Bulk All Schedules 12312012',
                     'FFIEC CDR Call Bulk All Schedules 03312013',
                     'FFIEC CDR Call Bulk All Schedules 06302013',
                     'FFIEC CDR Call Bulk All Schedules 09302013',
                     'FFIEC CDR Call Bulk All Schedules 12312013',
                     'FFIEC CDR Call Bulk All Schedules 03312014',
                     'FFIEC CDR Call Bulk All Schedules 06302014',
                     'FFIEC CDR Call Bulk All Schedules 09302014',
                     'FFIEC CDR Call Bulk All Schedules 12312014',
                     'FFIEC CDR Call Bulk All Schedules 03312015',
                     'FFIEC CDR Call Bulk All Schedules 06302015',
                     'FFIEC CDR Call Bulk All Schedules 09302015',
                     'FFIEC CDR Call Bulk All Schedules 12312015',
                     'FFIEC CDR Call Bulk All Schedules 03312016',
                     'FFIEC CDR Call Bulk All Schedules 06302016',
                     'FFIEC CDR Call Bulk All Schedules 09302016',
                     'FFIEC CDR Call Bulk All Schedules 12312016',
                     'FFIEC CDR Call Bulk All Schedules 03312017',
                     'FFIEC CDR Call Bulk All Schedules 06302017',
                     'FFIEC CDR Call Bulk All Schedules 09302017',
                     'FFIEC CDR Call Bulk All Schedules 12312017',
                     'FFIEC CDR Call Bulk All Schedules 03312018',
                     'FFIEC CDR Call Bulk All Schedules 06302018',
                     'FFIEC CDR Call Bulk All Schedules 09302018',
                     'FFIEC CDR Call Bulk All Schedules 12312018',
                     'FFIEC CDR Call Bulk All Schedules 03312019',
                     'FFIEC CDR Call Bulk All Schedules 06302019',
                     'FFIEC CDR Call Bulk All Schedules 09302019',
                     'FFIEC CDR Call Bulk All Schedules 12312019',
                     'FFIEC CDR Call Bulk All Schedules 03312020',
                     'FFIEC CDR Call Bulk All Schedules 06302020',
                     'FFIEC CDR Call Bulk All Schedules 09302020',
                     'FFIEC CDR Call Bulk All Schedules 12312020',
                     'FFIEC CDR Call Bulk All Schedules 03312021',
                     'FFIEC CDR Call Bulk All Schedules 06302021',
                     'FFIEC CDR Call Bulk All Schedules 09302021',
                     'FFIEC CDR Call Bulk All Schedules 12312021',
                     'FFIEC CDR Call Bulk All Schedules 03312022',
                     'FFIEC CDR Call Bulk All Schedules 06302022',
                     'FFIEC CDR Call Bulk All Schedules 09302022',
                     'FFIEC CDR Call Bulk All Schedules 12312022',
                     'FFIEC CDR Call Bulk All Schedules 03312023',
                     'FFIEC CDR Call Bulk All Schedules 06302023',
                     'FFIEC CDR Call Bulk All Schedules 09302023',
                     'FFIEC CDR Call Bulk All Schedules 12312023',
                     'FFIEC CDR Call Bulk All Schedules 03312024',
                     'FFIEC CDR Call Bulk All Schedules 06302024',
                     'FFIEC CDR Call Bulk All Schedules 09302024']
    
    array_claims_to_liquidity_all = []
    for file_name in file_names:
        date = file_name[-8:]
        file_path_data = os.path.join(data_path, file_name)
        #this datafile gives RCONF051 - Amount of deposit accounts (excluding retirement accounts) of more than $250,000
        # and RCONF047 - Amount of retirement deposit accounts of more than $250,000
        #total unisured deposits = RCONF047 + RCONF051
        print(file_path_data)
        #############################################Calculate Uninsured Deposits###################################################
        try:
            file_path_RCO  =  os.path.join(file_path_data, f'FFIEC CDR Call Schedule RCO {date}(1 of 2).txt')
            df_RCO = pd.read_csv(file_path_RCO, delimiter="\t", low_memory=False)
        except:
            file_path_RCO  =  os.path.join(file_path_data, f'FFIEC CDR Call Schedule RCO {date}.txt')
            df_RCO = pd.read_csv(file_path_RCO, delimiter="\t", low_memory=False)
        #Skip first row
        df_RCO = df_RCO.iloc[1:]
        df_RCO['RCONF051'] = pd.to_numeric(df_RCO['RCONF051'], errors='coerce')
        df_RCO['RCONF047'] = pd.to_numeric(df_RCO['RCONF047'], errors='coerce')
        
        df_RCO['Total_Uninsured_Deposits'] = df_RCO['RCONF051'] + df_RCO['RCONF047']
        #print(df_RCO['Total_Uninsured_Deposits'])
        
        #Read RCE 1 File
        #This contains RCONJ474 (RCE-I) - Unisured time deposits
        
        file_RCE1 = os.path.join(file_path_data, f'FFIEC CDR Call Schedule RCEI {date}.txt')
        df_RCE1 = pd.read_csv(file_RCE1, delimiter="\t", low_memory=False)
        df_RCE1 = df_RCE1.iloc[1:]
        df_RCE1['RCONJ474'] = pd.to_numeric(df_RCE1['RCONJ474'], errors='coerce')
        #print(df_RCE1['RCONJ474'])
        
        ##
        #Read file RCK 
        #This file contains - RCFN3404 (RC-K) - Interest-bearing deposits in foreign offices, EDGE and Agreement subsidiaries, and IBFs
        file_path_RCK = os.path.join(file_path_data, f'FFIEC CDR Call Schedule RCK {date}.txt')
        df_RCK = pd.read_csv(file_path_RCK, delimiter="\t", low_memory=False)
        df_RCK = df_RCK.iloc[1:]
        df_RCK['RCFN3404'] = pd.to_numeric(df_RCK['RCFN3404'], errors='coerce')
        #print(df_RCK['RCFN3404'])
        
        ##Credit Lines
        #Read file RCL 
        #This file contains - RCFDJ457 (RC-L) - Commercial and industrial loans - Credit lines
        file_path_RCL = os.path.join(file_path_data, f'FFIEC CDR Call Schedule RCL {date}(1 of 2).txt')
        df_RCL = pd.read_csv(file_path_RCL, delimiter="\t", low_memory=False)
        df_RCL = df_RCL.iloc[1:]
        df_RCL['RCFDJ457'] = pd.to_numeric(df_RCL['RCFDJ457'], errors='coerce')
        
        ##Reserves
        #Read file RCA -Reserves are from RCFD0090 (RCON0090 if missing),
        #This file contains - RCFDJ457 (RC-A) - Balances due from Federal Reserve Banks - Reserves
        file_path_RCA = os.path.join(file_path_data, f'FFIEC CDR Call Schedule RCA {date}.txt')
        df_RCA = pd.read_csv(file_path_RCA, delimiter="\t", low_memory=False)
        df_RCA = df_RCA.iloc[1:]
        df_RCA['RCFD0090'] = pd.to_numeric(df_RCA['RCFD0090'], errors='coerce')
        df_RCA['RCON0090'] = pd.to_numeric(df_RCA['RCON0090'], errors='coerce')
        
        #eligible assets 
        #Eligible assets consist of Treasury and Agency securities that were eligible for swap with the Fed for reserves in at
        #least one quantitative easing round between 2008Q4-2023Q1. 
        
        #Schedule RC-B of Call Reports (labelled as Eligible Assets for brevity) which is the sum of the banks’ 
        #holdings of US treasuries, obligations of US Government agencies,
        #securities issued by US States and Political Subdivisions, and agency-backed mortgage-backed securities
        
        #UST - RCFD0211,  RCFD0213, RCFD1286, RCFD1287
        #obligations of US Gov agencies - RCFDHT50,  RCFDHT51, RCFDHT52, RCFDHT53 (exists after 2018-06-30) (Do not need this )
        #Securities by US States and Subdivisions - RCFD8496,  RCFD8497, RCFD8498, RCFD8499 (Do not need this)
        
        #Agency backed MBS - 1. Guaranteed by GNMA - RCFDG300,  RCFDG301, RCFDG302, RCFDG303 (after 2009)
        #             2. Issued or guaranteed by U.S. Government agencies or sponsored agencies - RCFDG312, RCFDG313,RCFDG314, RCFDG315 (after 2009)
        #             3. Issued or guaranteed by FNMA, FHLMC, or GNMA - RCFDK142, RCFDK143, RCFDK144, RCFDK145 (exists after 2011-03-31)
        #             4. Issued or guaranteed by U.S. Government agencies or sponsored agencies - RCFDK150, RCFDK151, RCFDK152, RCFDK153 (after 2011-03-31)
        
        file_path_RCB = os.path.join(file_path_data, f'FFIEC CDR Call Schedule RCB {date}(1 of 2).txt')
        df_RCB = pd.read_csv(file_path_RCB, delimiter="\t", low_memory=False)
        df_RCB = df_RCB.iloc[1:]
        try:
            #try if all are available
            #UST
            columns_to_convert = [
                'RCFD0211', 'RCFD0213', 'RCFD1286', 'RCFD1287', 

                'RCFDG300', 'RCFDG301', 'RCFDG302', 'RCFDG303',
                'RCFDG312', 'RCFDG313', 'RCFDG314', 'RCFDG315',
                'RCFDK142', 'RCFDK143', 'RCFDK144', 'RCFDK145',
                'RCFDK150', 'RCFDK151', 'RCFDK152', 'RCFDK153']
            
            # Convert columns to numeric with coercion for invalid data
            df_RCB[columns_to_convert] = df_RCB[columns_to_convert].apply(pd.to_numeric, errors='coerce')
            
            # Compute the Eligible_Assets column as the sum of all specified columns
            df_RCB['Eligible_Assets'] = df_RCB[columns_to_convert].sum(axis=1)
        except:
 
            #remove post 2011
            #UST
            columns_to_convert = [
                'RCFD0211', 'RCFD0213', 'RCFD1286', 'RCFD1287',
                'RCFDG300', 'RCFDG301', 'RCFDG302', 'RCFDG303',
                'RCFDG312', 'RCFDG313', 'RCFDG314', 'RCFDG315']
            
            # Convert the specified columns to numeric with coercion
            df_RCB[columns_to_convert] = df_RCB[columns_to_convert].apply(pd.to_numeric, errors='coerce')
            
            # Optional: If you want to calculate the sum of these columns into a new column 'UST_Eligible_Assets'
            df_RCB['Eligible_Assets'] = df_RCB[columns_to_convert].sum(axis=1)
            
               
        
        df_merged_uninsured_1 = pd.merge(df_RCO, df_RCE1, on = 'IDRSSD', how = 'inner')
        df_merged_uninsured = pd.merge(df_merged_uninsured_1 , df_RCK, on = 'IDRSSD', how = 'inner')
        
        df_merged_uninsured['Uninsured_Demand_Deposits'] = ( df_merged_uninsured['Total_Uninsured_Deposits'].fillna(0) - 
            df_merged_uninsured['RCONJ474'].fillna(0) - df_merged_uninsured['RCFN3404'].fillna(0))

        df_merged_1 = pd.merge(df_merged_uninsured, df_RCA, on = 'IDRSSD', how = 'inner') #put reserves
        df_merged_2 = pd.merge(df_merged_1, df_RCL, on = 'IDRSSD', how = 'inner')#Put credit lines - RCFDJ457
        df_merged = pd.merge(df_merged_2, df_RCB, on = 'IDRSSD', how = 'inner') #put eligible assets
        
        #Claims toPotential Liquidity: (Credit Lines + Uninsured Demandable Deposits)/(Reserves + Eligible Assets)
        
        #print(df_merged_uninsured['Uninsured_Demand_Deposits'])
        uninsured_demand_deposits = df_merged['Uninsured_Demand_Deposits'].sum()
        credit_lines = df_merged['RCFDJ457'].sum()
        
        df_merged['reserves'] = df_merged.apply(lambda x: x['RCFD0090'] if pd.notna(x['RCFD0090']) else x['RCON0090'], axis=1)
        reserves = df_merged['reserves'].sum()
        eligible_assets = df_merged['Eligible_Assets'].sum()
        
        claim_to_liquidity = (credit_lines + uninsured_demand_deposits)/(reserves + eligible_assets)
        df_claims_to_liquidity = pd.DataFrame({
                'Date': [date],
                'claim_to_liquidity': [claim_to_liquidity],
                'Uninsured_Demand_Deposits': [uninsured_demand_deposits], 
                'credit_lines': [credit_lines], 
                'reserves': [reserves]
            })
        df_claims_to_liquidity['Date'] = pd.to_datetime(df_claims_to_liquidity['Date'], format='%m%d%Y')
        #print(df_uninsured_demand_deposits)
        #break
        array_claims_to_liquidity_all.append(df_claims_to_liquidity)
        #######################################################################################################################################
        
        #############################################
        
    df_claims_to_liquidity_all = pd.concat(array_claims_to_liquidity_all, ignore_index=True)
    
    #plt.plot(df_claims_to_liquidity_all['Date'], df_claims_to_liquidity_all['claim_to_liquidity'])
    #plt.show()
    return df_claims_to_liquidity_all

def correlation_with_SE_bootstrap(series_1, series_2):
    number_of_samples = 50000
    n = len(series_1)
    corr_array = []
    count = 0
    obs_diff = np.abs(series_1.corr(series_2) - 0)
    for ii in range(number_of_samples):

        random_sample_1 = np.random.choice(series_1, size=n, replace=True)
        random_sample_2 = np.random.choice(series_2, size=n, replace=True)
        corr = np.corrcoef(random_sample_1, random_sample_2)[0, 1]
        corr_array.append(corr)
        #p value
        diff = np.abs(corr)
        if diff >= obs_diff:
            count += 1
    p_value = count/number_of_samples
    corr_array = np.array(corr_array)
    #plt.plot(corr_array)
    return series_1.corr(series_2), np.std( corr_array), p_value
    
def figure_correlation(merged_data):
    
    # Plot
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Primary y-axis
    ax1.plot(merged_data['Date'], np.log(merged_data['claim_to_liquidity']), label='Bank Fragility', color='blue')
    #ax1.plot(merged_data['Date'], merged_data['reserves']*10**(-6), label='Reserves Call Report', color='green')
    ax1.set_ylabel('Bank Fragility', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(merged_data['Date'], np.log(merged_data['TOTRESNS']), label='Reserves', color='red')
    ax2.set_ylabel('Reserves', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Common features
    ax1.set_xlabel('Date')
    plt.title('Bank Fragility and Reserves Over Time')
    fig.tight_layout()
    
    # Legends
    ax1.legend(loc='upper left', bbox_to_anchor=(0.80, 1), fontsize=10)

    # Secondary y-axis legend (reserves), placed below the first one
    ax2.legend(loc='upper left', bbox_to_anchor=(0.80, 0.9), fontsize=10)
    plt.grid()
    
    correlation = merged_data['claim_to_liquidity'].corr(merged_data['TOTRESNS'])
    correlation, error, p_value = correlation_with_SE_bootstrap(merged_data['claim_to_liquidity'], merged_data['TOTRESNS'])
    correlation_text = f"$\\rho$: {correlation:.3f} ({error:.3f}) "
    ax1.text(0.01, 0.9, correlation_text, transform=ax1.transAxes, ha='left', va='top', fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=1'))
    fig.tight_layout()
    plt.show()
    
    

    #ax1.plot(merged_data['Date'], np.log(merged_data['claim_to_liquidity']), label='Bank Fragility', color='blue')
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plotting Bank Fragility (Primary Y-Axis)
    ax1.plot(
        merged_data['Date'], 
        np.log(merged_data['claim_to_liquidity'] / merged_data['claim_to_liquidity'].shift(1)), 
        label='Fragility', 
        color='green'
    )
    ax1.set_ylabel('Bank Fragility', color='green', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='green')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.axhline(0, color='green', linestyle='--', linewidth=0.8, alpha=0.7)
    # Create a secondary Y-axis for Reserves
    ax2 = ax1.twinx()
    ax2.plot(
        merged_data['Date'], 
        np.log(merged_data['TOTRESNS'] / merged_data['TOTRESNS'].shift(1)), 
        label='Reserves', 
        color='red'
    )
    ax2.set_ylabel('Reserves', color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.axhline(0, color='blue', linestyle='--', linewidth=0.8, alpha=0.7)
    # Add a title
    plt.title('Log Change in Bank Fragility and Reserves Over Time', fontsize=14)
    
    # Adding legends
    ax1.legend(loc='upper left', bbox_to_anchor=(0.8, 1), fontsize=10)
    ax2.legend(loc='upper left', bbox_to_anchor=(0.8, 0.9), fontsize=10)
    
    # Improve layout
    fig.tight_layout()
    
    # Add grid to the primary axis
    ax1.grid(visible=True, which='both', linestyle='--', linewidth=0.5)
    series_1 = np.log(merged_data['claim_to_liquidity'] / merged_data['claim_to_liquidity'].shift(1)).dropna()
    series_2 = np.log(merged_data['TOTRESNS'] / merged_data['TOTRESNS'].shift(1)).dropna()
    
    correlation = series_1.corr(series_2)
    correlation, error, p_value = correlation_with_SE_bootstrap(series_1, series_2)
    correlation_text = f"$\\rho$: {correlation:.3f} ({error:.3f}) "
    ax1.text(0.01, 0.9, correlation_text, transform=ax1.transAxes, ha='left', va='top', fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=1'))
    fig.tight_layout()
    plt.show()
    # Show the plot
    plt.show()
    #plt.savefig(os.path.join(base_path, 'correlation_reserves_fragility.png'))
def adf_test(series, name):
    result = adfuller(series)
    print(f"ADF Statistic for {name}: {result[0]}")
    print(f"P-value for {name}: {result[1]}")
    print(f"Critical Values for {name}: {result[4]}")
    
if __name__ == '__main__':
    #data = data_read()
    #data.to_csv(os.path.join(base_path, 'claims_to_liquidity.csv'))
    
    
    reserves = pd.read_csv('/Users/varuundeshpande/Downloads/TOTRESNS.csv')
    reserves['observation_date'] = pd.to_datetime(reserves['observation_date'])
    reserves = reserves[reserves['observation_date'].dt.year >= 2010]
    data = pd.read_csv(os.path.join(base_path, 'claims_to_liquidity.csv'))
    data['Date'] = pd.to_datetime(data['Date'])
    
    reserves['quarter'] = reserves['observation_date'].dt.to_period('Q')
    quarterly_reserves = reserves.groupby('quarter').last().reset_index()
    
    # Convert the bank fragility data to quarterly periods
    data['quarter'] = data['Date'].dt.to_period('Q')
    
    # Merge the datasets on the quarter column
    merged_data = pd.merge(data, quarterly_reserves, on='quarter', how='inner')
    
    #figure_correlation(merged_data)
    
    
    #figure_correlation(merged_data)
    
    reserves_normalized = (merged_data['TOTRESNS']-merged_data['TOTRESNS'].mean())/merged_data['TOTRESNS'].std()
    fragility_normalized = (merged_data['claim_to_liquidity']-merged_data['claim_to_liquidity'].mean())/merged_data['claim_to_liquidity'].std()
 
    #data = pd.DataFrame({'series_1': reserves_normalized, 'series_2': fragility_normalized})
    
    
    
    merged_data['reserves_change_log'] = np.log(merged_data['TOTRESNS']/merged_data['TOTRESNS'].shift(4))
    merged_data['deposits_change_log'] = np.log(merged_data['Uninsured_Demand_Deposits']/merged_data['Uninsured_Demand_Deposits'].shift(4))
    merged_data['credit_lines_change'] = np.log(merged_data['credit_lines']/merged_data['credit_lines'].shift(4))
    merged_data['claim_to_liquidity_change_log'] = np.log(merged_data['claim_to_liquidity']/merged_data['claim_to_liquidity'].shift(4))

    merged_data['reserves_lag_log']  = np.log(merged_data['TOTRESNS'].shift(4))
    
    X = merged_data[['reserves_change_log', 'reserves_lag_log']].dropna()  # Reserves (independent variable)
    Y = merged_data['claim_to_liquidity_change_log'].dropna()  # Bank Fragility (dependent variable)
    
    # Add a constant term to the independent variable
    X = sm.add_constant(X)
    
    # Fit the OLS regression model
    model = sm.OLS(Y, X).fit()
    
    
    # Print the regression results
    print(model.summary())
    
    merged_data['frag_lag_log']  = np.log(merged_data['claim_to_liquidity'].shift(4))
    X = merged_data[['claim_to_liquidity_change_log', 'frag_lag_log']].dropna()  # Reserves (independent variable)
    Y = merged_data['reserves_change_log'].dropna()  # Bank Fragility (dependent variable)
    
    # Add a constant term to the independent variable
    X = sm.add_constant(X)
    
    # Fit the OLS regression model
    model = sm.OLS(Y, X).fit()
    
    
    # Print the regression results
    print(model.summary())

        
        
        
        
                        