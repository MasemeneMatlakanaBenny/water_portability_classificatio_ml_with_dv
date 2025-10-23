import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from dvclive import Live


# load the train set
train_df=pd.read_csv("data/train_df.csv")


# get into the DVC experiment:
with Live(save_dvc_exp=True) as live:
    
    # define the X_train and y_train:
    X_train=train_df.drop("Potability",axis=1)
    y_train=train_df["Potability"]

    # fit the model -> Logistic Regression:
    model=LogisticRegression(solver="liblinear")

    model.fit(X_train,y_train)

    # now save the model: use pickle or json file 
    joblib.dump(model,"models/model_fitted.pkl")

    # get the train predictions and probabilities using the model:
    y_preds=model.predict(X_train)
    y_probs=model.predict_proba(X_train)[:,1]  

    # log the plots now:
    live.log_sklearn_plot("confusion_matrix",
                          y_train,
                          y_preds,
                          title="Confusion Matrix - train set",
                          name="train/conf_mat"
                        )
    

