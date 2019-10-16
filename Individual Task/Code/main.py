import os
import argparse
import tempfile
import shutil
import pandas as pd
import numpy as np
import datetime
from timeit import default_timer as timer
import sklearn as sk
from math import sqrt
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from sklearn.feature_extraction import FeatureHasher
import scipy
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats.mstats import gmean
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor

FILE_FORMAT = "csv"

FILES = {
    "submission": {
        "random": "tcd ml 2019-20 income prediction submission file example (random solutions).{}".format(FILE_FORMAT),
        "use": "tcd ml 2019-20 income prediction submission file.{}".format(FILE_FORMAT)
    },
    "test": "tcd ml 2019-20 income prediction test (without labels).{}".format(FILE_FORMAT),
    "training": "tcd ml 2019-20 income prediction training (with labels).{}".format(FILE_FORMAT)
}


def get_data(path):
    if not path:
        raise FileNotFoundError("{} not found".format(path))
    data = pd.read_csv(path)
    return data


def process_training(df, training):
    """
    To manipulate the training data
    :param df: pandas DataFrame
    :param training: Boolean if training is being performed or not - will print plot if training is True
    :return: matrix of features (X) [features_matrix], income (Y) [income], stats on training data [stats]
    """

    """
    1. Instance - Instance of data (irrelevant)

    2. Year of Record -  Year record was made - scale on geo mean
    3. Gender - Participant gender - encode -> (male) (female) (other) - "Bad"/nan -> (other)
    4. Age - Age of participant - scale on geo mean
    5. Country - Country participant is from/works in - one hot encode
    6. Size of City - Population size - on geo mean
    7. Profession - Participant job - label encoding
    8. University Degree - Level of education - encode -> (no) (bachelor) (master) (phd) - "Bad"/nan -> (no)
    9. Wears Glasses - Boolean glasses wearing - [Not used] 
    10. Hair Colour - Hair - one-hot encode from unique values -> "Bad"/nan -> (other)
    11. Body height [cm] - Tallness - scale on geo mean
    
    12. Income in EUR - income - leave as is - desired output
    """

    # if remove_negative:
    #     income = df["Income in EUR"].to_numpy(dtype=float) #.reshape(l, 1)
    #     # Remove negative incomes
    #     remove_indexes = list(df["Income in EUR"].where(lambda x: x < 0).dropna().index)
    #     df = df.drop(df.index[remove_indexes])
    # df = df[np.abs(scipy.stats.zscore(df["Income in EUR"])) < 2.7]
    # df = df[df["Income in EUR"] < 300000]
    # df = df[df["Age"] < 90]
    # df = df[np.abs(scipy.stats.zscore(df["Age"].where(lambda x: x > 0).dropna())) < 1.5]
    l = len(df)
    df_stats = df.describe(include="all")
    features_matrix = np.ones([l, 1])
    stats = dict()

    income = df["Income in EUR"].to_numpy(dtype=float).reshape(l, 1)

    # instance = df["Instance"].to_numpy(dtype=int)

    year_record = df["Year of Record"].to_numpy(dtype=int)
    year_record = pd.Series(year_record)
    f_year_record = year_record.where(lambda x: x > 0).dropna().to_numpy(dtype=int)
    year_geo_mean = int(gmean(f_year_record))
    stats.update({"year_mean": year_geo_mean})
    year_record = year_record.where(lambda x: x > 0, year_geo_mean).to_numpy(dtype=int)
    year_record_scaled = year_record / year_geo_mean
    features_matrix = np.append(features_matrix, year_record_scaled.reshape(l, 1), axis=1)
    if training:
        plot_relations(year_record, income, "Year of Record")

    gender = df["Gender"].fillna("other").str.lower() #.to_numpy(dtype=str)
    # Unique -> ['0' 'other' 'female' 'male' nan 'unknown']
    male = gender.where(lambda x: x.str.lower() == "male", 0).replace("male", 1).to_numpy(dtype=int)
    female = gender.where(lambda x: x.str.lower() == "female", 0).replace("female", 1).to_numpy(dtype=int)
    other = pd.Series(male + female).replace(1, 2).replace(0, 1).replace(2, 0).to_numpy(dtype=int)
    features_matrix = np.append(
        features_matrix, (np.append(
            male.reshape(l, 1),
            np.append(
                female.reshape(l, 1),
                other.reshape(l, 1),
                axis=1),
            axis=1)),
        axis=1
    )
    # if training:
    #     plot_relations(male, income, "Male")
    #     plot_relations(female, income, "Female")
    #     plot_relations(other, income, "Other Gender")

    age_df = df["Age"]
    filtered_age = age_df.where(lambda x: x > 0).dropna().to_numpy(dtype=int)
    # filtered_age = df[np.abs(
    #     scipy.stats.zscore(age_df.dropna().astype("int"))) < 1.5].to_numpy(dtype=int)
    gm_age = int(gmean(filtered_age))
    stats.update({"age_mean": gm_age})
    age = age_df.where(lambda x: x > 0, gm_age).to_numpy(dtype=int)
    age_scaled = age / gm_age
    features_matrix = np.append(features_matrix, age_scaled.reshape(l, 1), axis=1)
    if training:
        plot_relations(age, income, "Age")

    country = df["Country"].fillna("other").str.lower()
    # country_count_df = pd.DataFrame({"Country": country.value_counts().index, "Count": country.value_counts()})
    # top_countries = country_count_df["Country"][:100].tolist()
    # for i, c in enumerate(country):
    #     if any(x == c for x in top_countries):
    #         pass
    #     else:
    #         country[i] = "other"
    # hashed_countries = country.astype("category").cat.codes.to_numpy()
    # features_matrix = np.append(features_matrix, hashed_countries.reshape([l, 1]), axis=1)
    country_df = country.unique()
    country_list = sorted(country_df.tolist())
    stats.update({"country_list": country_list})
    one_hot_c = np.zeros([l, len(country_df)])
    for i, c in enumerate(country_list):
        one_hot_c[:, i] = country.where(lambda x: x == c, 0).replace(c, 1).to_numpy(dtype=int)
    features_matrix = np.append(features_matrix, one_hot_c, axis=1)

    population = df["Size of City"]
    pop_mean = int(gmean(population.where(lambda x: x > 0).dropna().to_numpy(dtype=int)))
    stats.update({"pop_mean": pop_mean})
    population = population.where(lambda x: x > 0).fillna(pop_mean).to_numpy(dtype=int) / pop_mean
    features_matrix = np.append(features_matrix, population.reshape(l, 1), axis=1)

    job_series = df["Profession"].fillna("other").str.lower()
    # hashed_jobs = sk.feature_extraction.FeatureHasher(
    #     n_features=int(1.5 * len(job_series.unique())), input_type="string"
    # ).transform(job_series).toarray()
    # features_matrix = np.append(features_matrix, hashed_jobs, axis=1)
    # pandas_categories = job_series.astype("category").cat.codes.to_numpy().reshape([l, 1])
    # features_matrix = np.append(features_matrix, pandas_categories, axis=1)
    le = preprocessing.LabelEncoder()
    stats.update({"job_encoder": le})
    encoded = le.fit_transform(job_series.tolist())
    features_matrix = np.append(features_matrix, encoded.reshape(l, 1), axis=1)
    # split_job = pd.Series(job_series.unique()).apply(lambda x: x.split(" ")).tolist()
    # job_adj_set = set()
    # for job_list in split_job:
    #     for job in job_list:
    #         job_adj_set.add(job)
    #     # for job in job_list:
    #     #     if not any(c in job for c in ("and", "the", "&", "an", "-")):
    #     #         job_adj_set.add(job)
    # job_adj_list = sorted(list(job_adj_set))
    # stats.update({"job_list": job_adj_list})
    # one_hot_j = np.zeros([l, len(job_adj_list)])
    # for i, adj in enumerate(job_adj_list):
    #     # one_hot_j[:, i] = job_series.where(lambda x: adj in x).fillna(0)
    #     # one_hot_j[:, i] = job_series.str.find(adj).replace(0, 1).replace(-1, 0).to_numpy(dtype=int)
    #     one_hot_j[:, i] = job_series.str.contains(adj).to_numpy(dtype=int)
    # features_matrix = np.append(features_matrix, one_hot_j, axis=1)

    degree = df["University Degree"].fillna("other").str.lower()
    # unq_degree = degree.unique()
    # Unique -> ['bachelor' 'master' 'phd' 'no' '0' nan]
    bachelor = degree.where(lambda x: x.str.lower() == "bachelor", 0).replace("bachelor", 1).to_numpy(dtype=int)
    master = degree.where(lambda x: x.str.lower() == "master", 0).replace("master", 1).to_numpy(dtype=int)
    phd = degree.where(lambda x: x.str.lower() == "phd", 0).replace("phd", 1).to_numpy(dtype=int)
    other = pd.Series(bachelor + master + phd).replace(1, 2).replace(0, 1).replace(2, 0).to_numpy(dtype=int)
    features_matrix = np.append(
        features_matrix, (np.append(
                bachelor.reshape(l, 1),
                np.append(
                    master.reshape(l, 1),
                    np.append(
                        phd.reshape(l, 1),
                        other.reshape(l, 1),
                        axis=1),
                    axis=1),
                axis=1)),
        axis=1
    )
    # bachelor = degree.where(lambda x: x.str.lower() == "bachelor", 0).replace("bachelor", 1).to_numpy(dtype=int)
    # master = degree.where(lambda x: x.str.lower() == "master", 0).replace("master", 2).to_numpy(dtype=int)
    # phd = degree.where(lambda x: x.str.lower() == "phd", 0).replace("phd", 3).to_numpy(dtype=int)
    # degree_encoded = bachelor + master + phd
    # features_matrix = np.append(features_matrix, degree_encoded.reshape(l, 1), axis=1)
    # plot_relations(degree_encoded, income, "Education")

    glasses = df["Wears Glasses"].fillna(0).to_numpy(dtype=int)
    # Unique -> [0 1]
    features_matrix = np.append(features_matrix, glasses.reshape(l, 1), axis=1)

    # hair = df["Hair Color"].fillna("other").str.lower()
    # # Unique -> ['Blond' 'Black' 'Brown' nan 'Red' 'Unknown' '0']
    # blond = hair.where(lambda x: x.str.lower() == "blond", 0).replace("blond", 1).to_numpy(dtype=int)
    # black = hair.where(lambda x: x.str.lower() == "black", 0).replace("black", 1).to_numpy(dtype=int)
    # brown = hair.where(lambda x: x.str.lower() == "brown", 0).replace("brown", 1).to_numpy(dtype=int)
    # red = hair.where(lambda x: x.str.lower() == "red", 0).replace("red", 1).to_numpy(dtype=int)
    # other = pd.Series(blond + black + brown + red).replace(1, 2).replace(0, 1).replace(2, 0).to_numpy(dtype=int)
    # features_matrix = np.append(
    #     features_matrix, (np.append(
    #             blond.reshape(l, 1),
    #             np.append(
    #                 black.reshape(l, 1),
    #                 np.append(
    #                     brown.reshape(l, 1),
    #                     np.append(
    #                         red.reshape(l, 1),
    #                         other.reshape(l, 1),
    #                         axis=1),
    #                     axis=1),
    #                 axis=1),
    #             axis=1)),
    #     axis=1
    # )

    height_np = df["Body Height [cm]"].fillna(0).to_numpy(dtype=int)
    height_df = pd.Series(height_np)
    gm_height = int(gmean(height_df.where(lambda x: x > 0).dropna().to_numpy(dtype=int)))
    stats.update({"height_mean": gm_height})
    height = height_df.where(lambda x: x > 0, gm_height).to_numpy(dtype=int) / gm_height
    features_matrix = np.append(features_matrix, height.reshape(l, 1), axis=1)

    return features_matrix, income, stats


def process_test(df, stats):
    """
    To manipulate the training data
    :param df: pandas DataFrame
    :param stats: dictionary containing statistics from training data to be applied on the test data
    :return: matrix of features (X) [features_matrix]
    """

    """
    1. Instance - Instance of data (irrelevant)

    2. Year of Record -  Year record was made - scale on geo mean
    3. Gender - Participant gender - encode -> (male) (female) (other) - "Bad"/nan -> (other)
    4. Age - Age of participant - scale on geo mean
    5. Country - Country participant is from/works in - one hot encode
    6. Size of City - Population size - on geo mean
    7. Profession - Participant job - label encoding
    8. University Degree - Level of education - encode -> (no) (bachelor) (master) (phd) - "Bad"/nan -> (no)
    9. Wears Glasses - Boolean glasses wearing - [Not used] 
    10. Hair Colour - Hair - one-hot encode from unique values -> "Bad"/nan -> (other)
    11. Body height [cm] - Tallness - scale on geo mean

    12. Income - income - leave as is - desired output
    """

    l = len(df)
    features_matrix = np.ones([l, 1])

    # instance = df["Instance"].to_numpy(dtype=int)

    year_record = df["Year of Record"].to_numpy(dtype=int)
    year_record = pd.Series(year_record)
    year_geo_mean = stats["year_mean"]
    year_record = year_record.where(lambda x: x > 0, year_geo_mean).to_numpy(dtype=int) / year_geo_mean
    features_matrix = np.append(features_matrix, year_record.reshape(l, 1), axis=1)

    gender = df["Gender"].fillna("other").str.lower() #.to_numpy(dtype=str)
    # Unique -> ['0' 'other' 'female' 'male' nan 'unknown']
    male = gender.where(lambda x: x.str.lower() == "male", 0).replace("male", 1).to_numpy(dtype=int)
    female = gender.where(lambda x: x.str.lower() == "female", 0).replace("female", 1).to_numpy(dtype=int)
    other = pd.Series(male + female).replace(1, 2).replace(0, 1).replace(2, 0).to_numpy(dtype=int)
    features_matrix = np.append(
        features_matrix, (np.append(
            male.reshape(l, 1),
            np.append(
                female.reshape(l, 1),
                other.reshape(l, 1),
                axis=1),
            axis=1)),
        axis=1
    )

    age_df = df["Age"]
    gm_age = stats["age_mean"]
    age = age_df.where(lambda x: x > 0, gm_age).to_numpy(dtype=int) / gm_age
    features_matrix = np.append(features_matrix, age.reshape(l, 1), axis=1)

    country = df["Country"].fillna("other").str.lower()
    country_list = stats["country_list"]
    one_hot_c = np.zeros([l, len(country_list)])
    for i, c in enumerate(country_list):
        one_hot_c[:, i] = country.where(lambda x: x == c, 0).replace(c, 1).to_numpy(dtype=int)
    features_matrix = np.append(features_matrix, one_hot_c, axis=1)

    population = df["Size of City"]
    pop_mean = stats["pop_mean"]
    population = population.where(lambda x: x > 0).fillna(pop_mean).to_numpy(dtype=int) / pop_mean
    features_matrix = np.append(features_matrix, population.reshape(l, 1), axis=1)

    job_series = df["Profession"].fillna("other").str.lower()
    # split_job = pd.Series(job_series.unique()).apply(lambda x: x.split(" ")).tolist()
    # job_adj_set = set()
    # for job_list in split_job:
    #     for job in job_list:
    #         job_adj_set.add(job)
    # job_adj_list = sorted(list(job_adj_set))
    # pandas_categories = job_series.astype("category").cat.codes.to_numpy().reshape([l, 1])
    # features_matrix = np.append(features_matrix, pandas_categories, axis=1)
    encoder = stats["job_encoder"]
    encoded = encoder.fit_transform(job_series.tolist())
    features_matrix = np.append(features_matrix, encoded.reshape(l, 1), axis=1)
    # job_adj_list = stats["job_list"]
    # one_hot_j = np.zeros([l, len(job_adj_list)])
    # for i, adj in enumerate(job_adj_list):
    #     # one_hot_j[:, i] = job_series.where(lambda x: adj in x).fillna(0)
    #     # one_hot_j[:, i] = job_series.str.find(adj).replace(0, 1).replace(-1, 0).to_numpy(dtype=int)
    #     one_hot_j[:, i] = job_series.str.contains(adj).to_numpy(dtype=int)
    # features_matrix = np.append(features_matrix, one_hot_j, axis=1)

    degree = df["University Degree"].fillna("other").str.lower()
    bachelor = degree.where(lambda x: x.str.lower() == "bachelor", 0).replace("bachelor", 1).to_numpy(dtype=int)
    master = degree.where(lambda x: x.str.lower() == "master", 0).replace("master", 1).to_numpy(dtype=int)
    phd = degree.where(lambda x: x.str.lower() == "phd", 0).replace("phd", 1).to_numpy(dtype=int)
    other = pd.Series(bachelor + master + phd).replace(1, 2).replace(0, 1).replace(2, 0).to_numpy(dtype=int)
    features_matrix = np.append(
        features_matrix, (np.append(
                bachelor.reshape(l, 1),
                np.append(
                    master.reshape(l, 1),
                    np.append(
                        phd.reshape(l, 1),
                        other.reshape(l, 1),
                        axis=1),
                    axis=1),
                axis=1)),
        axis=1
    )
    # bachelor = degree.where(lambda x: x.str.lower() == "bachelor", 0).replace("bachelor", 1).to_numpy(dtype=int)
    # master = degree.where(lambda x: x.str.lower() == "master", 0).replace("master", 2).to_numpy(dtype=int)
    # phd = degree.where(lambda x: x.str.lower() == "phd", 0).replace("phd", 3).to_numpy(dtype=int)
    # degree_encoded = bachelor + master + phd
    # features_matrix = np.append(features_matrix, degree_encoded.reshape(l, 1), axis=1)

    glasses = df["Wears Glasses"].fillna(0).to_numpy(dtype=int)
    # Unique -> [0 1]
    features_matrix = np.append(features_matrix, glasses.reshape(l, 1), axis=1)

    # hair = df["Hair Color"].fillna("other").str.lower()
    # # Unique -> ['Blond' 'Black' 'Brown' nan 'Red' 'Unknown' '0']
    # blond = hair.where(lambda x: x.str.lower() == "blond", 0).replace("blond", 1).to_numpy(dtype=int)
    # black = hair.where(lambda x: x.str.lower() == "black", 0).replace("black", 1).to_numpy(dtype=int)
    # brown = hair.where(lambda x: x.str.lower() == "brown", 0).replace("brown", 1).to_numpy(dtype=int)
    # red = hair.where(lambda x: x.str.lower() == "red", 0).replace("red", 1).to_numpy(dtype=int)
    # other = pd.Series(blond + black + brown + red).replace(1, 2).replace(0, 1).replace(2, 0).to_numpy(dtype=int)
    # features_matrix = np.append(
    #     features_matrix, (np.append(
    #             blond.reshape(l, 1),
    #             np.append(
    #                 black.reshape(l, 1),
    #                 np.append(
    #                     brown.reshape(l, 1),
    #                     np.append(
    #                         red.reshape(l, 1),
    #                         other.reshape(l, 1),
    #                         axis=1),
    #                     axis=1),
    #                 axis=1),
    #             axis=1)),
    #     axis=1
    # )

    height_np = df["Body Height [cm]"].fillna(0).to_numpy(dtype=int)
    height_df = pd.Series(height_np)
    gm_height = stats["height_mean"]
    height = height_df.where(lambda x: x > 0, gm_height).to_numpy(dtype=int) / gm_height
    features_matrix = np.append(features_matrix, height.reshape(l, 1), axis=1)

    return features_matrix


# def cleanup(return_dir, tmp_dir):
#     cwd = os.getcwd()
#     os.chdir(return_dir)
#     shutil.rmtree(cwd)
#     if os.path.exists(cwd):
#         raise FileExistsError("{} exists".format(tmp_dir))
#     os.chdir(return_dir)


def write_predictions(df, predictions, output_file, data_dir, tmp_dir):
    """
    Function to write predictions to submission files for submitting results to Kaggle
    :param df: DataFrame from submissions file to be written to
    :param predictions: Numpy array of predicted incomes
    :param output_file: Path to submission file
    :param data_dir: Path to directory containing original files
    :param tmp_dir: Path to directory containing modified files - to be modified
    :return:
    """
    print("Prediction stats:")
    print(pd.DataFrame(predictions).describe())
    # Write to test file
    df["Income"] = predictions
    df.to_csv(output_file, index=False)
    # Write to submission file
    os.chdir(data_dir)
    submission_file = shutil.copy(FILES["submission"]["use"], tmp_dir)
    os.chdir(tmp_dir)
    submission_df = get_data(submission_file)
    submission_df["Income"] = predictions
    submission_df.to_csv(submission_file, index=False)


def run(linear=True, training=True):
    """
    Main function used to process data, create models and output predictions
    :param linear: Boolean, if the prediction model is linear or not
    :param training: Boolean, if the model is being trained or being used to test output
    :return:
    """
    script_dir = os.getcwd()
    root_dir = os.path.dirname(script_dir)
    os.chdir(root_dir)
    data_dir = os.path.join(root_dir, "Data")

    tmp_dir = os.path.join(root_dir, "tmp")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    os.chdir(data_dir)
    training_file = shutil.copy(FILES.get("training"), tmp_dir)
    os.chdir(tmp_dir)
    training_data = get_data(training_file)
    x, y, stats = process_training(training_data, training)

    if linear:
        re_model = linear_model.Ridge(alpha=0.1, normalize=False, fit_intercept=False, tol=1e-5)
        # re_model = linear_model.RidgeCV(fit_intercept=False)
        # re_model = linear_model.Lasso()
        # re_model = linear_model.LinearRegression()
        # poly = PolynomialFeatures(degree=2)
        # X_ = poly.fit_transform(x)
    else:
        re_model = CatBoostRegressor(learning_rate=0.08, iterations=20000, task_type="GPU", use_best_model=True)
        # re_model = RandomForestRegressor(n_estimators='warn',
        #                                  criterion="mse",
        #                                  max_depth=None,
        #                                  min_samples_split=2,
        #                                  min_samples_leaf=1,
        #                                  min_weight_fraction_leaf=0.,
        #                                  max_features="auto",
        #                                  max_leaf_nodes=None,
        #                                  min_impurity_decrease=0.,
        #                                  min_impurity_split=None,
        #                                  bootstrap=True,
        #                                  oob_score=False,
        #                                  n_jobs=None,
        #                                  random_state=None,
        #                                  verbose=0,
        #                                  warm_start=False)

    x_train, x_val, y_train, y_val = model_selection.train_test_split(x, y, test_size=0.2, random_state=1)

    # Use training data
    print("Training")
    start = timer()
    if linear:
        if training:
            re_model.fit(x_train, y_train)
        else:
            re_model.fit(x, y)
    else:
        re_model.fit(x_train, y=y_train.flatten(), use_best_model=True, eval_set=(x_val, y_val.flatten()),
                     verbose=10000)
    end = timer()
    dur = end - start
    print("Training took: {}".format(str(datetime.timedelta(seconds=dur))))
    if not linear or (linear and training):
        if linear:
            print('Coefficients: \n', re_model.coef_)
        y_val_pred = re_model.predict(x_val)
        mse = sk.metrics.mean_squared_error(y_val, y_val_pred)
        rmse = sqrt(mse)
        print("RMSE: {:,.5f}".format(rmse))
        print('Variance score: %.2f' % sk.metrics.r2_score(y_val, y_val_pred))

        # Prediction stats
        print("Validation stats:")
        print(pd.DataFrame(y_val_pred).describe())

    # Output test predictions
    os.chdir(data_dir)
    test_file = shutil.copy(FILES.get("test"), tmp_dir)
    os.chdir(tmp_dir)

    test_data = get_data(test_file)
    x_test = process_test(test_data, stats)
    y_test_pred = re_model.predict(x_test)

    write_predictions(test_data, y_test_pred, test_file, data_dir, tmp_dir)

    os.chdir(script_dir)


def feature_correlations(test_df, correlation_feature="Income in EUR"):
    """
    Function for finding correlations between features
    :param test_df: Pandas DataFrame containing all data for correlation checking
    :param correlation_feature: Feature to compare correlation against, a column of the input DataFrame
    :return: The resultant correlations
    """
    df = test_df.copy()
    columns = list(df.columns.values)
    for column in columns:
        if column != correlation_feature:
            df[column] = df[column].astype("category").cat.codes

    correlations = df[df.columns[1:]].corr()[correlation_feature][:]
    print(correlations)

    return correlations


def plot_relations(x, y, x_label, y_label="Income"):
    """
    Function to plot features to (roughly) visualise correlation (using a scatter plot)
    :param x: x feature
    :param y: y feature
    :param x_label: label for x feature
    :param y_label: label for y feature
    :return:
    """
    plt.figure()
    x, y = list(zip(*sorted(list(zip(x, y.flatten())), key=lambda x: x[0])))
    plt.scatter(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-l", "--linear", dest="linear", action="store_true",
                        help="Boolean - the model in use is linear",
                        default=False)
    parser.add_argument("-no-l", "--no-linear", dest="linear", action="store_false",
                        help="Boolean - the model in use is not linear",
                        default=False)
    parser.add_argument("-t", "--training", dest="training", action="store_true",
                        help="Boolean - model is being trained - will generate plots",
                        default=False)
    parser.add_argument("-no-t", "--no-training", dest="training", action="store_false",
                        help="Boolean - model is not being trained - no plots",
                        default=False)

    args = parser.parse_args()

    run(linear=args.linear, training=args.training)
