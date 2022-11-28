import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_df():
    df = pd.read_csv("V2_ENG_interview_dataset.csv", dtype=str, infer_datetime_format=True)
    df = df.loc[df["Unnamed: 10"].isna()]
    df = df.dropna(axis=1)
    pd.to_numeric(df["Amount"], errors="raise");
    types = {"Year-Month":int, "Agency Number":int,	"Agency Name":str, "Cardholder Last Name":str, "Cardholder First Initial":str, "Amount":float, "Vendor":str, "Transaction Date":str, "Posted Date":str,	"Merchant Category Code (MCC)":str}
    df = df.astype(types)
    df["Posted Date"] = pd.to_datetime(df["Posted Date"])
    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"])
    df["Days Delay"] = (df["Posted Date"] - df["Transaction Date"]).dt.days
    df.insert(1, "Year", df["Year-Month"]//100)
    df.insert(2, "Month", df["Year-Month"]%100)
    df.drop("Year-Month", axis=1, inplace=True)
    df_op = df.loc[df["Amount"] > 0]
    return df_op


def get_person_balanced_df(df, person, categories, cat_vars, num_vars):
    transations_list = []
    labels_list = []
    df_cp = df[categories].copy()
    cat_data = pd.get_dummies(df_cp[cat_vars])
    numeric_data = df_cp[num_vars].copy()
    numeric_cat_data = pd.concat([numeric_data, cat_data], axis=1)
    initials = df_cp.loc[df['Cardholder Last Name'] == person]['Cardholder First Initial'].value_counts().index.to_list()

    for initial in initials:
        numeric_cat_label_data = numeric_cat_data.copy()
        numeric_cat_label_data['label'] = (df['Cardholder Last Name'] == person)*(df['Cardholder First Initial'] == initial)
        df_true_transations = numeric_cat_label_data.loc[numeric_cat_label_data['label'] == True].copy()
        df_false_transations = numeric_cat_label_data.loc[numeric_cat_label_data['label'] == False].sample(n=df_true_transations.shape[0]).copy()
        df_transations = pd.concat([df_true_transations, df_false_transations], axis=0)
        label = df_transations['label']
        df_transations.drop('label', axis=1, inplace=True)
        if df_transations.shape[0] > 100:
            transations_list.append(df_transations)
            labels_list.append(label)
    return transations_list, labels_list


def get_person_true_transations_df(df, person, categories, cat_vars, num_vars):
    transations_list = []
    labels_list = []
    df_cp = df[categories].copy()
    cat_data = pd.get_dummies(df_cp[cat_vars])
    numeric_data = df_cp[num_vars].copy()
    numeric_cat_data = pd.concat([numeric_data, cat_data], axis=1)
    initials = df_cp.loc[df['Cardholder Last Name'] == person]['Cardholder First Initial'].value_counts().index.to_list()

    for initial in initials:
        numeric_cat_label_data = numeric_cat_data.copy()
        numeric_cat_label_data['label'] = (df['Cardholder Last Name'] == person)*(df['Cardholder First Initial'] == initial)
        df_transations = numeric_cat_label_data.loc[numeric_cat_label_data['label'] == True].copy()
        label = df_transations['label']
        df_transations.drop('label', axis=1, inplace=True)
        if df_transations.shape[0] > 50:
            transations_list.append(df_transations)
            labels_list.append(label)
    return transations_list, labels_list

def get_person_total_df(df, person, categories, cat_vars, num_vars):
    transations_list = []
    labels_list = []
    df_cp = df[categories].copy()
    cat_data = pd.get_dummies(df_cp[cat_vars])
    numeric_data = df_cp[num_vars].copy()
    numeric_cat_data = pd.concat([numeric_data, cat_data], axis=1)
    initials = df_cp.loc[df['Cardholder Last Name'] == person]['Cardholder First Initial'].value_counts().index.to_list()

    for initial in initials:
        df_transations = numeric_cat_data.copy()
        label = (df['Cardholder Last Name'] == person)*(df['Cardholder First Initial'] == initial)
        if df_transations.shape[0] > 100:
            transations_list.append(df_transations)
            labels_list.append(label)
    return transations_list, labels_list

def plot_confusion_matrix(cm, target_names, title='Confusion Matrix', cmap=plt.cm.Greens):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()

    width, height = cm.shape

    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
