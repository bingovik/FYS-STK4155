# Remove outliers
df_out = df.copy()

data_todrop1 = (df_out[(df_out['LIMIT_BAL'] > 800000)])
data_todrop2 = df_out.loc[:, 'BILL_AMT1':'BILL_AMT6'][((df_out.loc[:, 'BILL_AMT1':'BILL_AMT6'] < 0) |(df_out.loc[:, 'BILL_AMT1':'BILL_AMT6'] > 700000)).any(axis=1)]
data_todrop3 = df_out.loc[:, 'PAY_AMT1':'PAY_AMT6'][((df_out.loc[:, 'PAY_AMT1':'PAY_AMT6'] < 0) | (df_out.loc[:, 'PAY_AMT1':'PAY_AMT6'] > 400000)).any(axis=1)]

data_todrop1 = data_todrop1.to_numpy()
data_todrop2 = data_todrop2.index.values
data_todrop3 = data_todrop3.index.values
data_todrop = np.concatenate((data_todrop1, data_todrop2, data_todrop3), axis=0)
data_todrop = np.unique(data_todrop)

df_out.drop(data_todrop, inplace = True)
