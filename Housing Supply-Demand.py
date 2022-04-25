#!/usr/bin/env python
# coding: utf-8

# # HOUSING PRICES KEY SUPPLY-DEMAND FACTORS
# ### BY KONARK PAHUJA 21/04/22

# ### IMPORTS

# In[724]:


import numpy as np 
import pandas as pd 
import seaborn as sb
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
from functools import reduce
import requests 


# In[725]:


sb.set_style("whitegrid")


# ### FUNCTIONS

# In[726]:


def cleanup(df, col_name):
    df['DATE'] = pd.to_datetime(df["DATE"])
    df['year'] = df['DATE'].dt.year
    df = df.groupby("year", as_index=True).mean()
    df.rename(columns = {list(df)[0]: col_name}, inplace =True)    
    return df


# In[727]:


def cleanupIncome(df):
    #df = df[0]
    cols_to_drop=[0,1,2]
    df.drop(columns=df.columns[cols_to_drop],inplace=True)
    df.dropna(inplace=True, axis=0)
    df.drop([0,1], inplace=True)
    df.reset_index(drop=True, inplace=True)
    #return df
        


# In[728]:


def cleanupEdu(df):
    #df = df[0]
    cols_to_drop=[0,1,3]
    df.drop(columns=df.columns[cols_to_drop],inplace=True)
    df.dropna(inplace=True, axis=0)
    df.drop([0,1], inplace=True)
    df.reset_index(drop=True, inplace=True)
    #return df
        


# In[729]:


def cleanupFam(df):
    #df = df[0]
    cols_to_drop=[0,1]
    df.drop(columns=df.columns[cols_to_drop],inplace=True)
    df.dropna(inplace=True, axis=0)
    df.drop([0,1], inplace=True)
    df.reset_index(drop=True, inplace=True)
    #return df        


# # HOUSING PRICES

# ### LOAD DATA

# In[730]:


#source: https://fred.stlouisfed.org/

#S&P/Case-Shiller U.S. National Home Price Index (CSUSHPISA)
housing_prices = pd.read_csv("Data/CSUSHPISA.csv")

#Homeownership Rate in the United States (RSAHORUSQ156S)
home_ownership = pd.read_csv("Data/RSAHORUSQ156S.csv")


# In[731]:


housing_prices.head()


# ### CLEAN DATA

# In[732]:


housing_prices = cleanup(housing_prices,"median housing price")
home_ownership = cleanup(home_ownership,"home ownership %")


# In[733]:


housing_prices


# ### VISUALISE

# In[734]:



figure,ax = plt.subplots(2,1,constrained_layout=True,figsize=(14,8),dpi=300) 


#homeownership rate
sb.lineplot(ax=ax[0],data = own_price, x="year", y="home ownership %", linewidth = 2,color='r')
ax[0].set_title("homeownership rate")
ax[0].set(xlabel="year", ylabel = "%")
ax[0].set_xticks(np.arange(2002, 2023, 1.0)) 


#housing prices
sb.lineplot(ax=ax[1],data = own_price, x="year", y="median housing price", linewidth = 2)
ax[1].set_title("housing price")
ax[1].set(xlabel="year", ylabel = "y2000=100")
ax[1].set_xticks(np.arange(2002, 2023, 1.0)) 


# # SUPPLY FACTORS:

# ## 1. PRODUCTION

# ### 1.1 CONSTRUCTION STAGES

# ### LOAD DATA

# In[735]:


#STAGES OF CONSTRUCTION:

#source: https://fred.stlouisfed.org/

#Completed (NHFSEPCS) units *1000

df_stages_completed = pd.read_csv("Data/Supply/Stages/NHFSEPCS.csv")

#Under Construction (NHFSEPUCS)

df_stages_under_const = pd.read_csv("Data/Supply/Stages/NHFSEPUCS.csv")

#Not Started (NHFSEPNTS)

df_stages_not_start = pd.read_csv("Data/Supply/Stages/NHFSEPNTS.csv")


# In[736]:


df_stages_completed


# ### CLEAN DATA

# In[737]:


df_stages_completed = cleanup(df_stages_completed,"completed")
df_stages_under_const = cleanup(df_stages_under_const,"under construction")
df_stages_not_start = cleanup(df_stages_not_start, "not started")


# In[738]:


df_supply_house_stages = pd.concat([df_stages_completed, df_stages_under_const,df_stages_not_start]
                                   ,axis=1)


# In[739]:


df_supply_house_stages


# ### VISUALISE:

# In[740]:


figure,ax = plt.subplots(2,1,constrained_layout=True,figsize=(14,8),dpi=300) 

#homes for sale by construction  
sb.lineplot(ax= ax[0],data = df_supply_house_stages,
            linewidth = 2)
ax[0].set(xlabel="year", ylabel = "no. units for sale * 1000")
ax[0].set_title("HOMES FOR SALE BY STAGE OF CONSTRUCTION (US, LAST 20 YEARS)")
ax[0].set_xticks(np.arange(2002, 2023, 1.0))


#housing prices
sb.lineplot(ax=ax[1],data = housing_prices, x="year", y="median housing price", linewidth = 2)
ax[1].set_title("HOUSING PRICE")
ax[1].set(xlabel="year", ylabel = "y2000=100")
ax[1].set_xticks(np.arange(2002, 2023, 1.0)) 


# ### 1.2 CONSTRUCTION COSTS

# ### LOAD DATA

# In[741]:


#STAGES OF CONSTRUCTION: 

#source: https://fred.stlouisfed.org/

#Producer Price Index by Commodity: Special Indexes: Construction Materials (WPUSI012011)

df_cost_mat = pd.read_csv("Data/Supply/Costs/WPUSI012011.csv")

# Producer Price Index by Industry: Construction Machinery Manufacturing (PCU333120333120)

df_cost_mach = pd.read_csv("Data/Supply/Costs/PCU333120333120.csv")

# Import Price Index (End Use): Crude Oil (IR10000)

df_cost_oil = pd.read_csv("Data/Supply/Costs/IR10000.csv")

#Import Price Index (End Use): Natural Gas (IR10110)

df_cost_gas = pd.read_csv("Data/Supply/Costs/IR10110.csv")

#All Employees, Residential Building (CES2023610001)

df_cost_labour = pd.read_csv("Data/Supply/Costs/CES2023610001.csv")


# In[742]:


df_cost_mat


# ### CLEAN DATA

# In[743]:


df_cost_mat = cleanup(df_cost_mat,"price: construction materials")
df_cost_mach = cleanup(df_cost_mach, "price: construction machinery")
df_cost_oil = cleanup(df_cost_oil,"price: crude oil import")
df_cost_gas = cleanup(df_cost_gas,"price: natural gas import")
df_cost_labour = cleanup(df_cost_labour, "employees: residential construct.")


# In[744]:


df_supply_costs = pd.concat([df_cost_mat,df_cost_mach,df_cost_oil,df_cost_gas,df_cost_labour], axis=1)


# In[745]:


df_supply_costs2 = pd.concat([df_cost_mat,df_cost_mach,df_cost_oil,df_cost_gas,df_cost_labour,housing_prices], axis=1)


# In[746]:


#normalize
df_supply_costs_norm =(df_supply_costs-df_supply_costs.min())/(df_supply_costs.max()-df_supply_costs.min())


# In[747]:


df_supply_costs_norm


# In[748]:


figure,ax = plt.subplots(2,1,constrained_layout=True,figsize=(14,8),dpi=300) 

#housing construction costs  
sb.lineplot(ax= ax[0],data = df_supply_costs_norm,
            linewidth = 2)
ax[0].set(xlabel="year", ylabel = "costs & labour normalized")
ax[0].set_title("COSTS AND LABOUR AVAILABILITY FOR HOUSING CONSTRUCTION")
ax[0].set_xticks(np.arange(2002, 2023, 1.0))
ax[0].legend(loc='best', bbox_to_anchor=(1, 1))

#housing prices
sb.lineplot(ax=ax[1],data = housing_prices, x="year", y="median housing price", linewidth = 2)
ax[1].set_title("HOUSING PRICE")
ax[1].set(xlabel="year", ylabel = "y2000=100")
ax[1].set_xticks(np.arange(2002, 2023, 1.0)) 


# # 2. POLICY:

# ## 2.1 ZONING

# ### LOAD DATA

# In[753]:


#source: https://fred.stlouisfed.org/

#New Privately-Owned Housing Units Authorized in Permit-Issuing Places: Single-Family Units (PERMIT1)

df_zone_1 = pd.read_csv("Data/Supply/zone/PERMIT1.csv")

#New Privately-Owned Housing Units Authorized in Permit-Issuing Places: Units in Buildings with 2-4 Units (PERMIT24)

df_zone_24 = pd.read_csv("Data/Supply/zone/PERMIT24.csv")

#New Privately-Owned Housing Units Authorized in Permit-Issuing Places: Units in Buildings with 5 Units or More (PERMIT5)

df_zone_5 = pd.read_csv("Data/Supply/zone/PERMIT5.csv")

# New Privately-Owned Housing Units Authorized in Permit-Issuing Places: Total Units (PERMIT)	

df_zone_tot = pd.read_csv("Data/Supply/zone/PERMIT.csv")


# In[750]:


df_zone_1


# ### CLEAN DATA

# In[30]:


df_zone_1 = cleanup(df_zone_1,"single family unit")
df_zone_24 = cleanup(df_zone_24,"2-4 units")
df_zone_5 = cleanup(df_zone_5, ">5 units")
df_zone_tot = cleanup(df_zone_tot, "total units")


# In[31]:


merge_dfs = [df_zone_1, df_zone_24,df_zone_5,df_zone_tot]


# In[530]:


df_zoning = pd.concat(merge_dfs,axis=1)


# In[751]:


df_zoning


# ### VISUALISE

# In[752]:


figure,ax = plt.subplots(2,1,constrained_layout=True,figsize=(14,8),dpi=300) 

#housing zoning  
sb.lineplot(ax= ax[0],data = df_zoning,
            linewidth = 2)
ax[0].set(xlabel="year", ylabel = "no. of units")
ax[0].set_title("HOUSING PERMITS AUTHORISED")
ax[0].set_xticks(np.arange(2002, 2023, 1.0))
ax[0].legend(loc='best', bbox_to_anchor=(1, 1))

#housing prices
sb.lineplot(ax=ax[1],data = housing_prices, x="year", y="median housing price", linewidth = 2)
ax[1].set_title("HOUSING PRICE")
ax[1].set(xlabel="year", ylabel = "y2000=100")
ax[1].set_xticks(np.arange(2002, 2023, 1.0)) 


# # 3. MARKET

# ### LOAD DATA

# In[758]:


#Data from Redfin

# Data for new vs existing sale nos. 2012-2022:
df_mark_sales = pd.read_csv("Data/Supply/Market/sales.csv", encoding='utf-16', sep= '\t')

#Data for new vs existing sale median prices 2012-2022:
df_mark_prices = pd.read_csv("Data/Supply/Market/median price.csv", encoding='utf-16', sep= '\t')


# In[759]:


df_mark_prices


# ### CLEAN DATA

# In[760]:


df_mark_sales.drop(["Region","Measure Names"], axis=1,inplace=True)
df_mark_prices.drop(["Region","Measure Names"], axis=1,inplace=True)


# In[761]:


df_mark_sales['New Construction Sales'] = None 
df_mark_sales['Existing House Sales'] = None

df_mark_prices['New Construction Median Sale Price'] = None 
df_mark_prices['Existing House Median Sale Price'] = None


# In[762]:


df_mark_sales["New Construction Sales"] = df_mark_sales.loc[0::2, "Measure Values"]
df_mark_sales["Existing House Sales"] = df_mark_sales.loc[1::2, "Measure Values"]
df_mark_sales.drop(columns=["Is New Construction Transaction","Measure Values"], inplace=True)

df_mark_prices["New Construction Median Sale Price"] = df_mark_prices.loc[0::2, "Measure Values"]
df_mark_prices["Existing House Median Sale Price"] = df_mark_prices.loc[1::2, "Measure Values"]
df_mark_prices.drop(columns=["Is New Construction Transaction","Measure Values"], inplace=True)


# In[763]:


df_mark_sales["Existing House Sales"] = df_mark_sales['Existing House Sales'].shift(-1)
df_mark_prices["Existing House Median Sale Price"] = df_mark_prices['Existing House Median Sale Price'].shift(-1)


# In[764]:


df_mark_sales.dropna(axis=0,inplace=True)
df_mark_prices.dropna(axis=0,inplace=True)


# In[765]:


df_mark_sales['Period End'] = pd.to_datetime(df_mark_sales["Period End"])
df_mark_sales['year'] = df_mark_sales['Period End'].dt.year
df_mark_sales = df_mark_sales.groupby("year", as_index=True).mean()

df_mark_prices['Period End'] = pd.to_datetime(df_mark_prices["Period End"])
df_mark_prices['year'] = df_mark_prices['Period End'].dt.year
df_mark_prices= df_mark_prices.groupby("year", as_index=True).mean()


# In[766]:


df_mark_sales


# ### VISUALISE

# In[769]:


figure,ax = plt.subplots(2,1,constrained_layout=True,figsize=(14,8),dpi=300) 

#housing market  
sb.lineplot(ax= ax[0],data = df_mark_sales,
            linewidth = 2)
ax[0].set(xlabel="year", ylabel = "no. of units")
ax[0].set_title("HOUSING MARKET BY HOUSE SALES, EXISTING VS NEW")
ax[0].set_xticks(np.arange(2002, 2023, 1.0))
ax[0].legend(loc='best', bbox_to_anchor=(1, 1))

#housing prices
sb.lineplot(ax=ax[1],data = housing_prices, x="year", y="median housing price", linewidth = 2)
ax[1].set_title("HOUSING PRICE")
ax[1].set(xlabel="year", ylabel = "y2000=100")
ax[1].set_xticks(np.arange(2002, 2023, 1.0)) 


# In[770]:


figure,ax = plt.subplots(2,1,constrained_layout=True,figsize=(14,8),dpi=300) 

#housing market  
sb.lineplot(ax= ax[0],data = df_mark_prices,
            linewidth = 2)
ax[0].set(xlabel="year", ylabel = "price of units")
ax[0].set_title("HOUSING MARKET BY HOUSE PRICES, EXISTING VS NEW")
ax[0].set_xticks(np.arange(2002, 2023, 1.0))
ax[0].legend(loc='best', bbox_to_anchor=(1, 1))

#housing prices
sb.lineplot(ax=ax[1],data = housing_prices, x="year", y="median housing price", linewidth = 2)
ax[1].set_title("HOUSING PRICE")
ax[1].set(xlabel="year", ylabel = "y2000=100")
ax[1].set_xticks(np.arange(2002, 2023, 1.0)) 


# # DEMAND FACTORS:

# ## 1. ECONOMY 

# ### LOAD DATA

# In[771]:


#source: https://fred.stlouisfed.org/


#30-Year Fixed Rate Mortgage Average in the United States (MORTGAGE30US) %

df_eco_mort = pd.read_csv("Data/Demand/Economy/MORTGAGE30US.csv")


#Consumer Price Index for All Urban Consumers: Owners' Equivalent Rent of Residences in U.S. City Average (CUSR0000SEHC)

df_eco_rent_eqv = pd.read_csv("Data/Demand/Economy/CUSR0000SEHC.csv")


#Consumer Price Index for All Urban Consumers: Rent of Primary Residence in U.S. City Average (CUSR0000SEHA)

df_eco_rent = pd.read_csv("Data/Demand/Economy/CUSR0000SEHA.csv")


#Real Gross Domestic Product (GDPC1): Billions of chained USD

df_eco_gdp = pd.read_csv("Data/Demand/Economy/GDPC1.csv")

#Inflation, consumer prices for the United States (FPCPITOTLZGUSA) %

df_eco_inf = pd.read_csv("Data/Demand/Economy/FPCPITOTLZGUSA.csv")

#Unemployment Rate (UNRATE)

df_eco_unemp = pd.read_csv("Data/Demand/Economy/UNRATE.csv")


# In[772]:


df_eco_mort


# ### CLEAN DATA

# In[781]:


df_eco_mort = cleanup(df_eco_mort,"30y fixed mortgage %")
df_eco_rent_eqv = cleanup(df_eco_rent_eqv,"owner's rent equivalent")
df_eco_rent = cleanup(df_eco_rent, "rent")
df_eco_gdp = cleanup(df_eco_gdp, "real gdp")
df_eco_inf = cleanup(df_eco_inf,"inflation %")
df_eco_unemp = cleanup(df_eco_unemp,"unemployment %")


# In[782]:


df_economy_rent = pd.concat([df_eco_rent,df_eco_rent_eqv],axis=1)


# In[783]:


df_economy_rates = pd.concat([df_eco_mort,df_eco_inf,df_eco_unemp],axis=1)


# In[773]:


df_economy_rates


# ### VISUALISE

# In[774]:


figure,ax = plt.subplots(2,1,constrained_layout=True,figsize=(14,8),dpi=300) 

#economy_rent: 
sb.lineplot(ax= ax[0],data = df_economy_rent,
            linewidth = 2)
ax[0].set(xlabel="year", ylabel = "cost index")
ax[0].set_title("RENT VS OWNERSHIP COST OF HOUSE")
ax[0].set_xticks(np.arange(2002, 2023, 1.0))
ax[0].legend(loc='best', bbox_to_anchor=(1, 1))

#housing prices
sb.lineplot(ax=ax[1],data = housing_prices, x="year", y="median housing price", linewidth = 2)
ax[1].set_title("HOUSING PRICE")
ax[1].set(xlabel="year", ylabel = "y2000=100")
ax[1].set_xticks(np.arange(2002, 2023, 1.0)) 


# In[775]:


figure,ax = plt.subplots(2,1,constrained_layout=True,figsize=(14,8),dpi=300) 

#economy_rent: 
sb.lineplot(ax= ax[0],data = df_economy_rates,
            linewidth = 2)
ax[0].set(xlabel="year", ylabel = "%")
ax[0].set_title("ECONOMY (Mortgage,Inflation,Unemployment)")
ax[0].set_xticks(np.arange(2002, 2023, 1.0))
ax[0].legend(loc='best', bbox_to_anchor=(1, 1))

#housing prices
sb.lineplot(ax=ax[1],data = housing_prices, x="year", y="median housing price", linewidth = 2)
ax[1].set_title("HOUSING PRICE")
ax[1].set(xlabel="year", ylabel = "y2000=100")
ax[1].set_xticks(np.arange(2002, 2023, 1.0)) 


# In[785]:


figure,ax = plt.subplots(2,1,constrained_layout=True,figsize=(14,8),dpi=300) 

#economy_rent: 
sb.lineplot(ax= ax[0],data = df_eco_gdp,
            linewidth = 2)
ax[0].set(xlabel="year", ylabel = "billions of chained USD 2012")
ax[0].set_title("ECONOMY (GDP)")
ax[0].set_xticks(np.arange(2002, 2023, 1.0))
ax[0].legend(loc='best', bbox_to_anchor=(1, 1))

#housing prices
sb.lineplot(ax=ax[1],data = housing_prices, x="year", y="median housing price", linewidth = 2)
ax[1].set_title("HOUSING PRICE")
ax[1].set(xlabel="year", ylabel = "y2000=100")
ax[1].set_xticks(np.arange(2002, 2023, 1.0)) 


# ## 2. DEMOGRAPHICS - HOUSEHOLD

# ### *Data availability : 2008-2012*

# ### 2.1 INCOME 

# ### LOAD DATA

# In[790]:


#source: CENSUSACS: ACS PUMS: https://data.census.gov/mdat/#/

hh_inc_list = []
for i in range(2008,2020):
    globals()["hh_inc_%s" % i] = pd.read_html("Data/Demand/Demo/Buyer/income/"+str(i)+".html")[0]
    hh_inc_list.append(globals()["hh_inc_%s" % i])


# ### CLEAN DATA

# In[791]:


columns_hh_inc = ['<50,000','50,001-100,000','100,001-150,000','150,001-200,000','200,001-500,000','>500,000']
df_demo_hh_inc = pd.DataFrame(index=range(2008,2020,1), columns = columns_hh_inc)
df_demo_hh_inc.index.rename('year', inplace=True)


# In[792]:


for item in hh_inc_list:
    cleanupIncome(item)


# In[793]:


for i in range(0,12):
    df_demo_hh_inc.iloc[i,:] = hh_inc_list[i].iloc[1]


# In[794]:


df_demo_hh_inc = df_demo_hh_inc.astype(float)


# In[795]:


df_demo_hh_inc


# ### VISUALISE

# In[796]:


figure,ax = plt.subplots(2,1,constrained_layout=True,figsize=(14,8),dpi=300) 

#buyer_demographic_household_income: 
sb.lineplot(ax= ax[0],data = df_demo_hh_inc)
ax[0].set(xlabel="year", ylabel = "No. of home buyers")
ax[0].set_title("Homebuyer Income")
ax[0].set_xticks(np.arange(2002, 2023, 1.0))
ax[0].legend(loc='best', bbox_to_anchor=(1, 1))

#housing prices
sb.lineplot(ax=ax[1],data = housing_prices, x="year", y="median housing price", linewidth = 2)
ax[1].set_title("HOUSING PRICE")
ax[1].set(xlabel="year", ylabel = "y2000=100")
ax[1].set_xticks(np.arange(2002, 2023, 1.0)) 


# ### 2.2 EDUCATION 

# ### LOAD DATA

# In[797]:


hh_edu_list = []
for i in range(2008,2020):
    globals()["hh_edu_%s" % i] = pd.read_html("Data/Demand/Demo/Buyer/education/"+str(i)+".html")[0]
    hh_edu_list.append(globals()["hh_edu_%s" % i])


# ### CLEAN DATA

# In[798]:


columns_hh_edu = ['Less than HighSchool','High School/GED',"Associate's deg","Bachelor's deg",
                  "Doctorate","Master's/Professional"]
df_demo_hh_edu = pd.DataFrame(index=range(2008,2020,1), columns = columns_hh_edu)
df_demo_hh_edu.index.rename('year', inplace=True)


# In[799]:


for item in hh_edu_list:
    cleanupEdu(item)


# In[800]:


for i in range(0,12):
    df_demo_hh_edu.iloc[i,:] = hh_edu_list[i].iloc[1]


# In[801]:


df_demo_hh_edu = df_demo_hh_edu.astype(float)


# In[802]:


df_demo_hh_edu


# ### VISUALISE

# In[803]:


figure,ax = plt.subplots(2,1,constrained_layout=True,figsize=(14,8),dpi=300) 

#buyer_demographic_household_education: 
sb.lineplot(ax= ax[0],data = df_demo_hh_edu)
ax[0].set(xlabel="year", ylabel = "No. of home buyers")
ax[0].set_title("Homebuyer Education")
ax[0].set_xticks(np.arange(2002, 2023, 1.0))
ax[0].legend(loc='best', bbox_to_anchor=(1, 1))

#housing prices
sb.lineplot(ax=ax[1],data = housing_prices, x="year", y="median housing price", linewidth = 2)
ax[1].set_title("HOUSING PRICE")
ax[1].set(xlabel="year", ylabel = "y2000=100")
ax[1].set_xticks(np.arange(2002, 2023, 1.0)) 


# ### 2.3 FAMILY TYPE

# ### LOAD DATA:

# In[805]:


hh_fam_list = []
for i in range(2008,2020):
    globals()["hh_fam_%s" % i] = pd.read_html("Data/Demand/Demo/Buyer/family type/"+str(i)+".html")[0]
    hh_fam_list.append(globals()["hh_fam_%s" % i])


# ### CLEAN DATA:

# In[806]:


columns_hh_fam= ["married couple", "male householder no spouse","female householder no spouse",
                 "male householder living alone","female householder living alone",
                 "male householder not living alone","female householder not living alone"]
df_demo_hh_fam = pd.DataFrame(index=range(2008,2020,1), columns = columns_hh_fam)
df_demo_hh_fam.index.rename('year', inplace=True)


# In[807]:


for item in hh_fam_list:
    cleanupFam(item)


# In[808]:


for i in range(0,12):
    df_demo_hh_fam.iloc[i,:] = hh_fam_list[i].iloc[1]


# In[809]:


df_demo_hh_fam = df_demo_hh_fam.astype(float)


# In[810]:


df_demo_hh_fam


# ### VISUALISE:

# In[811]:


figure,ax = plt.subplots(2,1,constrained_layout=True,figsize=(14,8),dpi=300) 

#buyer_demographic_household_education: 
sb.lineplot(ax= ax[0],data = df_demo_hh_fam)
ax[0].set(xlabel="year", ylabel = "No. of home buyers")
ax[0].set_title("Homebuyer Family Type")
ax[0].set_xticks(np.arange(2002, 2023, 1.0))
ax[0].legend(loc='best', bbox_to_anchor=(1, 1))

#housing prices
sb.lineplot(ax=ax[1],data = housing_prices, x="year", y="median housing price", linewidth = 2)
ax[1].set_title("HOUSING PRICE")
ax[1].set(xlabel="year", ylabel = "y2000=100")
ax[1].set_xticks(np.arange(2002, 2023, 1.0)) 


# In[813]:


#END OF PROJECT

