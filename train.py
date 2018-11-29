"""
Train LightGBM thar predicts Rossmann Sales, do exploratory
and post model analysis
"""
import pandas as pd
import datetime as dt
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
import multiprocessing
import shap
import cPickle as pickle
from sklearn.model_selection import GridSearchCV
import numpy as np
import yaml
import lightgbm as lgb

# Config files
# Load cross validation parameters
with open("model_parameters.config",
          "r") as f:
    cross_validation_parameters = yaml.load(f)
target = cross_validation_parameters['TARGET']
nbin = cross_validation_parameters['NBIN']
date_feature = cross_validation_parameters['DATE_FEATURE']
nbin_month = cross_validation_parameters['NBIN_MONTH']
nbin_bivariate = cross_validation_parameters['NBIN_BIVARIATE']
dev_size = cross_validation_parameters['DEV_SIZE']
test_size = cross_validation_parameters['TEST_SIZE']
seed = cross_validation_parameters['SEED']
n_estimators = cross_validation_parameters['N_ESTIMATORS']
learning_rate = cross_validation_parameters['LEARNING_RATE']
max_depth = cross_validation_parameters['MAX_DEPTH']
tree_method = cross_validation_parameters['TREE_METHOD']
subsample = cross_validation_parameters['SUBSAMPLE']
eval_metric = cross_validation_parameters['EVAL_METRIC']
early_stopping = cross_validation_parameters['EARLY_STOPPING']
business_features_keys = cross_validation_parameters['BUSINESS_FEATURES_KEYS']


def get_model_information(results_train, results_test,
                          y_prediction, y_test):
    """
    Get prediction distributions and learning curves
    """
    pdf_info = PdfPages('regression_model_information.pdf')
    # Score distribution
    plt.figure()
    plt.hist(y_prediction, normed=1, label="Prediction", alpha=0.5, bins=25)
    plt.hist(y_test, normed=1, label="Real", alpha=0.5, bins=25)
    plt.legend(loc='best')
    plt.grid()
    plt.title("Score distribution test set")
    pdf_info.savefig()
    plt.close()

    plt.figure()
    plt.plot(results_train, label="Train")
    plt.plot(results_test, label="Test")
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel("Trees")
    plt.ylabel(eval_metric)
    plt.title("Learning curves")
    pdf_info.savefig()
    plt.close()
    plt.close()
    pdf_info.close()


def from_letter_number(store):
    if store == "a":
        return 1
    elif store == "b":
        return 2
    elif store == "c":
        return 3
    elif store == "d":
        return 4
    else:
        return store


def promo_interval(promo):
    if promo == "Jan,Apr,Jul,Oct":
        return 1
    elif promo == "Feb,May,Aug,Nov":
        return 2
    elif promo == "Mar,Jun,Sept,Dec":
        return 3
    else:
        return promo


def shapear(x_test, target):
    # Open model
    reduced_df = x_test.sample(frac=0.25)
    with open(target + '_model.pkl', 'rb') as f:
        model = pickle.load(f)
    shap.initjs()
    shap_values = shap.TreeExplainer(model).shap_values(reduced_df)
    global_shap_vals = np.abs(shap_values).mean(0)
    global_shap_std = np.abs(shap_values).std(0)
    df = pd.DataFrame()
    df['features'] = reduced_df.columns
    df['shap'] = global_shap_vals
    df['shap_std'] = global_shap_std
    df = df.sort_values('shap', ascending=False)
    df.index = range(len(df))
    df.to_csv('shaps.csv')

    # Summary plot
    pdf_shap = PdfPages(target + '_shap.pdf')
    top_inds = np.argsort(-np.sum(np.abs(shap_values), 0))
    for i in top_inds:
        plt.figure()
        shap.dependence_plot(i, shap_values,
                             reduced_df, show=False,
                             interaction_index=None,
                             alpha=0.1)
        pdf_shap.savefig()
        plt.close()
    pdf_shap.close()
    return


def fill_nas(df, val=-1):
    """
    Fill NAs of a data frame with the value 'val', except for the
    column payment_delayed_time (which is NA if the loan has not been
    payed yet
    """
    # check the amount of NAs in each variable
    nas = df.isnull().mean()
    nas.sort_values(inplace=True)
    # fill NAs
    df_filled = df.ix[:, df.columns != 'payment_delayed_time'].fillna(val)
    if 'payment_delayed_time' in df.columns:
        df_filled['payment_delayed_time'] = df['payment_delayed_time']
    return df_filled



def time_analysis(df):
    month = df.groupby('Month').mean()
    month_keys = list(month.keys())
    pdf = PdfPages("month_average.pdf")
    for i in month_keys:
        y = list(month[i])
        x = list(month.index)
        plt.figure()
        plt.plot(x, y)
        plt.grid()
        plt.title("Month")
        plt.ylabel(i)
        plt.tick_params(axis='x', which='major', labelsize=6)
        pdf.savefig()
    pdf.close()


def bivariate(df):
    """
    Individual relationship between features and target
    """
    numeric = ['Customers', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'PromoInterval', 'Day_month']
    categorical = ['DayOfWeek', 'Promo', 'SchoolHoliday', 'StoreType', 'Assortment', 'Promo2', 'PromoInterval',
                   'StateHoliday', 'Promo_shift-1', 'Promo_shift+1', 'Week', 'Days_in_promo']
    pdf = PdfPages(target + '_bivariate.pdf')
    for i in df.keys():
        if i in numeric:
            try:
                if i != target:
                    flag_frame = df[[target, i]]
                    flag_frame['tile'] = pd.qcut(flag_frame[i].rank(method='first'), nbin_bivariate,
                                                 labels=range(1, nbin_bivariate + 1))
                    grouped = flag_frame.groupby('tile').mean()
                    x = list(grouped[i])
                    y = list(grouped[target])
                    plt.figure()
                    plt.plot(x, y, marker="o")
                    plt.grid(True)
                    plt.title(i)
                    plt.ylabel(target)
                    pdf.savefig()
                    plt.close()
            except:
                continue
        elif i in categorical:
            flag_frame = df[[target, i]]
            grouped = flag_frame.groupby(i).mean()
            x = list(grouped.index)
            y = list(grouped[target])

            plt.figure()
            plt.bar(x, y, align="center")
            plt.title(i)
            plt.ylabel(target)
            plt.grid()
            pdf.savefig()
            plt.close()
    pdf.close()


def train(df):
    X = df.copy()
    train, test = train_test_split(X, test_size=dev_size, random_state=seed)

    x_train = train[[i for i in X.keys() if i != target and i not in
                     business_features_keys]]
    x_test = test[[i for i in X.keys() if i != target and i not in
                   business_features_keys]]

    # Reorder
    x_train = x_train.reindex_axis(sorted(x_train.keys()), axis=1)
    x_test = x_test.reindex_axis(sorted(x_test.keys()), axis=1)

    # Save test frame for business metrics
    business_metrics_frame = test[business_features_keys]

    # Delete business features
    y_train = train[target]
    y_test = test[target]
    eval_set = [(x_train, y_train),
                (x_test, y_test)]
    model = lgb.LGBMRegressor(n_estimators=500000,
                              max_depth=5,
                              learning_rate=0.05,
                              seed=seed,
                              nthread=multiprocessing.cpu_count(),
                              subsample=0.7,
                              objective="rmse")

    model.fit(x_train, y_train,
              eval_metric="rmse",
              eval_set=eval_set,
              verbose=True,
              early_stopping_rounds=early_stopping,
              )
    results = model.evals_result_
    with open("columns.config", 'w') as \
             yaml_file:
         yaml.dump(dict(
             cols=list(x_train.columns)
         ), yaml_file, default_flow_style=False)
    # Save in pickle
    with open(target + '_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Shap
    shapear(x_test, target)

    # Feature importance
    importance_lst = model.feature_importances_
    importance_df = pd.DataFrame()
    importance_df['features'] = x_train.keys()
    importance_df['importance'] = importance_lst
    importance_df = importance_df.sort_values('importance', ascending=False)
    importance_df.to_csv(target + '_importance.csv', index=False)

    # Save pdf metrics
    results_train = results['training'][eval_metric]
    results_dev = results['valid_1'][eval_metric]
    y_prediction = model.predict(x_test)
    y_test = test[target]

    score = model.best_score_
    eval_metrics = dict(
        eval_metric=score,
        correlation=float(np.corrcoef(y_prediction, y_test)[1][0])
    )
    with open("regression_optimal_eval_metric.config", 'w') as \
            yaml_file:
        yaml.dump(eval_metrics, yaml_file, default_flow_style=False)

    get_model_information(results_train, results_dev, y_prediction, y_test)

    # Partial dependence plot
    x_partial_dependence = x_test.copy()
    x_partial_dependence[target + '_prediction'] = y_prediction
    x_partial_dependence[target] = test[target]
    x_partial_dependence[target] = y_test
    x_partial_dependence['predict_bin'] = pd.qcut(x_partial_dependence[
                                                  target + '_prediction'].
                                                  rank(method='first'), nbin,
                                                  labels=range(1, nbin + 1))
    # Verify monthly calibration per DLN
    x_partial_dependence[date_feature] = business_metrics_frame[
        date_feature]
    monthly_calibration(x_partial_dependence)
    calibration(x_partial_dependence)
    store_daily_calibration(x_partial_dependence)

def calibration(x_partial_dependence):
    """
    Calibration plot
    """
    x = x_partial_dependence.groupby('predict_bin').mean()[target]
    y = x_partial_dependence.groupby('predict_bin').mean()[target + "_prediction"]
    pdf_info = PdfPages('calibration.pdf')

    plt.figure()
    plt.plot(x, y, marker="o", alpha=0.5)
    plt.plot(y, y, alpha=0.5)
    plt.xlabel("Predictions")
    plt.ylabel("Real")
    plt.title("Calibration plot")
    plt.grid()
    pdf_info.savefig()
    plt.close()

    plt.figure()
    plt.scatter(x_partial_dependence['Sales_prediction'],
                x_partial_dependence['Sales'],alpha=0.2)
    plt.xlabel("Predictions")
    plt.ylabel("Real")
    plt.title("Predicted vs Real \n" + str(round(np.corrcoef(x_partial_dependence['Sales'],
                                                             x_partial_dependence['Sales_prediction'])[0][1], 3)))
    plt.grid()
    pdf_info.savefig()
    plt.close()
    pdf_info.close()



def store_daily_calibration(x_partial_dependence):
    """
    Get model's monthly calibration
    """
    # Calibration per month
    pdf = PdfPages('store_daily_calibration.pdf')
    stores = np.unique(x_partial_dependence['Store'])
    x_partial_dependence1 = x_partial_dependence[['Store', 'Month', 'Sales_prediction']]
    for i in stores:
        flag_frame1 = x_partial_dependence1[x_partial_dependence1['Store'] == i]
        grouped = flag_frame1.groupby('Month').mean()
        x = list(grouped.index)
        y = list(grouped['Sales_prediction'])
        real = list(grouped['Sales'])
        plt.figure()
        plt.plot(x, y, marker="o")
        plt.plot(x, real, alpha=0.6)
        plt.grid()
        plt.xlabel("Day")
        plt.tick_params(axis='x', which='major', labelsize=6)
        plt.ylabel("Predicted Sales")
        pdf.savefig()
        plt.close()
    pdf.close()







def monthly_calibration(x_partial_dependence):
    """
    Get model's monthly calibration
    """
    # Calibration per month
    months = np.unique(x_partial_dependence[date_feature])
    profit_month_df = pd.DataFrame()
    for month in months:
        flag_frame = x_partial_dependence[x_partial_dependence[
                                          date_feature] == month]
        flag_frame = flag_frame[[target + '_prediction', target,
                                 date_feature]]
        flag_frame['predict_bin'] = pd.qcut(flag_frame[
                                            target + '_prediction'].
                                            rank(method='first'), nbin_month,
                                            labels=range(1, nbin_month + 1))
        profit_month_df = profit_month_df.append(flag_frame)

    bins = np.unique(profit_month_df['predict_bin'])

    # Plot calibration per bin
    pdf = PdfPages('monthly_calibration.pdf')
    plt.figure()
    for tile in bins:
        flag_frame = profit_month_df[profit_month_df['predict_bin'] == tile]
        grouped = flag_frame.groupby(date_feature).mean()
        plt.plot(grouped[target + '_prediction'], marker="o",
                 label="Prediction")
        plt.plot(grouped[target], alpha=0.5, label="Real")
    # Title and other things
    plt.title("Monthly calibration")
    plt.ylabel(target)
    plt.tick_params(axis='x', which='major', labelsize=5)
    plt.grid()
    pdf.savefig()
    plt.close()
    pdf.close()



def clean_data(name, is_test):
    train = pd.read_csv(name + ".csv")
    store = pd.read_csv("store.csv")

    df = pd.merge(train, store, on="Store")
    df = df.sort_values('Date')
    if is_test == 0:
        df = df[df.Open.apply(lambda x: x == 1)]
        del df['Open']
        df = df[df['Sales'] > 0]
    df['Date'] = df.Date.apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d"))
    df['Month_nb'] = df.Date.apply(lambda x: x.month)
    df['Day_month'] = df.Date.apply(lambda x: x.day)
    df['Week'] = df.Date.apply(lambda x: x.week)
    df['Quarter'] = df.Date.apply(lambda x: x.quarter)

    df['Month'] = df['Date'].apply(
        lambda x: x.replace(day=1))
    df['StoreType'] = df.StoreType.apply(lambda x: from_letter_number(x))
    df['Assortment'] = df.Assortment.apply(lambda x: from_letter_number(x))
    df['PromoInterval'] = df.PromoInterval.apply(lambda x: promo_interval(x))
    df['StateHoliday'] = df.StateHoliday.apply(lambda x: float(from_letter_number(x)))

    df = df.sort_values(by=['Store', 'Date'], ascending=True, na_position='last')
    df.ix[(df['Promo'] == 1) & (df['Promo'].shift(1) == 0), 'PromoFirstDate'] = 1

    df['Year'] = df.Date.apply(lambda x: x.year)
    df['Years_With_Competition'] = df['Year'] - df['CompetitionOpenSinceYear']
    df['Years_with_Promo2'] = df['Year'] - df['Promo2SinceYear']
    del df['CompetitionOpenSinceYear']
    del df['Year']
    del df['Promo2SinceYear']

    return df


def test():
    test = clean_data("test", is_test=True)
    test = test.sort_values("Id")
    with open(target + '_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('columns.config', 'rb') as f:
        columns = yaml.load(f)['cols']

    df_test = test.copy()
    testdf = df_test[columns]
    y_prediction = [i for i in model.predict(testdf)]

    df_to_kaggle = pd.DataFrame(test['Id'])
    df_to_kaggle['Sales'] = y_prediction
    df_to_kaggle['Sales'][test['Open'] <= 0] = 0
    df_to_kaggle.to_csv('sample.csv', index=False)

    df_date = test[['Date', 'Store']]
    df_date['predictions'] = df_to_kaggle['Sales']
    stores = np.unique(df_date.Store)
    pdf = PdfPages('sales_prediction.pdf')
    for i in stores:
        flag_frame = df_date[df_date['Store'] == i]
        flag_frame = flag_frame.sort_values('Date')
        plt.figure()
        plt.plot(flag_frame.Date, flag_frame.predictions, marker="o",
                 label="Prediction")
        plt.grid()
        plt.xlabel("Day")
        plt.tick_params(axis='x', which='major', labelsize=6)
        plt.ylabel("Predicted Sales")
        pdf.savefig()
        plt.close()
    pdf.close()





def main():
    df = clean_data("train", is_test=False)
    time_analysis(df)
    bivariate(df)
    train(df)
    test()

if __name__ == "__main__":
    main()

