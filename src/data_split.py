import pandas as pd
from sklearn.model_selection import train_test_split

#read the data with pd.read_csv
df=pd.read_csv("data/water_potability.csv")

df=df.dropna()

# now split the dataset into both train and test sets:
train_df,test_df=train_test_split(df,test_size=0.2,random_state=42)

# save the train and test sets -> make sure that index has been set to false:
train_df.to_csv("data/train_df.csv",index=False)
test_df.to_csv("data/test_df.csv",index=False)

