import os
import tempfile
import shutil
import pandas as pd
import numpy as np
import sklearn as sk
from math import sqrt
from sklearn import linear_model
from sklearn import model_selection
import seaborn as sns
from sklearn.feature_extraction import FeatureHasher
import scipy
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats.mstats import gmean
import matplotlib.pyplot as plt

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


def process_training(df):
    """
    To manipulate the training data
    :param df: pandas dataframe
    :return:
    """

    """
    1. Instance - Instance of data (irrelevant)

    2. Year of Record -  Year record was made (possibly unimportant) - scale on geo mean
    3. Gender - Participant gender (relevant) - encode -> (male) (female) (other) - "Bad" -> (other)
    4. Age - Age of participant (relevant) - scale on min/max - remove top ages, too old
    5. Country - Country participant is from/works in (relevant) - use ISO country code ? -- one hot encode
    6. Size of City - Population size (somewhat relevant) - scale from ISO ? Aggregate with country in some way ?  -- encode per 10,000 ?
    7. Profession - Participant job (relevant) - try to categorise (use salary as metric) ? - word encoding
    8. University Degree - Level of education - encode -> (no) (bachelor) (master) (phd) - "Bad" -> (no)
    9. Wears Glasses - Boolean glasses wearing (possibly unimportant, age correlated ?) - encode (already ?) -> (no) (yes) - "Bad" -> (no)
    10. Hair Colour - Hair (possible relevant) - encode ? RGB ? - "Bad" -> bald ?
    11. Body height [cm] - Tallness (relevant) - scale to meters? - over 240 highly unlikely, ignore, skim top 5% ?

    12. Income in EUR - income - leas as is - desired output - remove negative incomes
    :return:
    """
    # income = df.iloc[:, 12]
    # if remove_negative:
    #     income = df["Income in EUR"].to_numpy(dtype=float) #.reshape(l, 1)
    #     # Remove negative incomes
    #     remove_indexes = list(df["Income in EUR"].where(lambda x: x < 0).dropna().index)
    #     df = df.drop(df.index[remove_indexes])
    # df = df[np.abs(scipy.stats.zscore(df["Income in EUR"])) < 1.8]
    # df = df[np.abs(scipy.stats.zscore(df["Age"].where(lambda x: x > 0).dropna())) < 1.5]
    l = len(df)
    df_stats = df.describe(include="all")
    features_matrix = np.ones([l, 1])
    stats = dict()

    income = df["Income in EUR"].to_numpy(dtype=float).reshape(l, 1)

    # instance = df.iloc[:, 0]
    instance = df["Instance"].to_numpy(dtype=int)
    # plt.figure()
    # plt.scatter(instance, instance)
    # plt.xlabel("Instance")
    # plt.ylabel("Instance")
    # plt.show()

    # year_record = df.iloc[:, 1]
    year_record = df["Year of Record"].to_numpy(dtype=int)
    # Unique ->
    year_record = pd.Series(year_record)
    f_year_record = year_record.where(lambda x: x > 0).dropna().to_numpy(dtype=int)
    # n_year_record = year_record.where(lambda x: x <= 0).to_numpy(dtype=int)
    # max_year = max(f_year_record)
    # min_year = min(f_year_record)
    year_geo_mean = int(gmean(f_year_record))
    stats.update({"year_mean": year_geo_mean})
    year_record = year_record.where(lambda x: x > 0, year_geo_mean).to_numpy(dtype=int)
    year_record_scaled = year_record / year_geo_mean
    features_matrix = np.append(features_matrix, year_record_scaled.reshape(l, 1), axis=1)
    plot_relations(year_record, income, "Year of Record")

    # gender = df.iloc[:, 2]
    gender = df["Gender"].str.lower() #.to_numpy(dtype=str)
    # Unique -> ['0' 'other' 'female' 'male' nan 'unknown']
    male = gender.where(lambda x: x.str.lower() == "male", 0).replace("male", 1).to_numpy(dtype=int)
    plot_relations(male, income, "Male")
    female = gender.where(lambda x: x.str.lower() == "female", 0).replace("female", 1).to_numpy(dtype=int)
    plot_relations(female, income, "Female")
    other = pd.Series(male + female).replace(1, 2).replace(0, 1).replace(2, 0).to_numpy(dtype=int)
    plot_relations(other, income, "Other Gender")
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

    # age = df.iloc[:, 3]
    age_df = df["Age"]
    filtered_age = age_df.where(lambda x: x > 0).dropna().to_numpy(dtype=int)
    # filtered_age = df[np.abs(
    #     scipy.stats.zscore(age_df.dropna().astype("int"))) < 1.5].to_numpy(dtype=int)
    gm_age = int(gmean(filtered_age))
    stats.update({"age_mean": gm_age})
    age = age_df.where(lambda x: x > 0, gm_age).to_numpy(dtype=int)
    age_scaled = age / gm_age
    features_matrix = np.append(features_matrix, age_scaled.reshape(l, 1), axis=1)
    plot_relations(age, income, "Age")

    # country = df.iloc[:, 4]
    country = df["Country"].str.lower()
    # Unique ->
    # Laos -> Lao
    # North Korea -> KOR
    # country_df = country.replace(
    #         "Laos", "LAO").replace(
    #         "South Korea", "KOR").replace(
    #         "North Korea", "PRK").replace(
    #         "DR Congo", "COD").unique()
    country_df = country.unique()
    country_list = sorted(country_df.tolist())
    stats.update({"country_list": country_list})
    one_hot_c = np.zeros([l, len(country_df)])
    for i, c in enumerate(country_list):
        one_hot_c[:, i] = country.where(lambda x: x == c, 0).replace(c, 1).to_numpy(dtype=int)
    features_matrix = np.append(features_matrix, one_hot_c, axis=1)
    # searched_countries = pd.Series(
    #     country.replace(
    #         "Laos", "LAO").replace(
    #         "South Korea", "KOR").replace(
    #         "North Korea", "PRK").replace(
    #         "DR Congo", "COD").unique()
    # ).apply(pyc.countries.search_fuzzy).tolist()

    # population = df.iloc[:, 5]
    population = df["Size of City"]
    pop_mean = int(gmean(population.where(lambda x: x > 0).dropna().to_numpy(dtype=int)))
    stats.update({"pop_mean": pop_mean})
    population = population.where(lambda x: x > 0).fillna(pop_mean).to_numpy(dtype=int) / pop_mean
    # max_pop = int(population.max())
    # min_pop = int(population.min())
    features_matrix = np.append(features_matrix, population.reshape(l, 1), axis=1)

    # job = df.iloc[:, 6]
    job_series = df["Profession"].str.lower().fillna("other")
    # hashed_jobs = sk.feature_extraction.FeatureHasher(
    #     n_features=int(1.5 * len(job_series.unique())), input_type="string"
    # ).transform(job_series).toarray()
    # features_matrix = np.append(features_matrix, hashed_jobs, axis=1)
    # pandas_categories = job_series.astype("category").cat.codes.to_numpy().reshape([l, 1])
    # features_matrix = np.append(features_matrix, pandas_categories, axis=1)
    split_job = pd.Series(job_series.unique()).apply(lambda x: x.split(" ")).tolist()
    job_adj_set = set()
    for job_list in split_job:
        for job in job_list:
            if len(job) > 2:
                job_adj_set.add(job)
    job_adj_list = sorted(list(job_adj_set))
    stats.update({"job_list": job_adj_list})
    one_hot_j = np.zeros([l, len(job_adj_list)])
    for i, adj in enumerate(job_adj_list):
        # one_hot_j[:, i] = job_series.where(lambda x: adj in x).fillna(0)
        # one_hot_j[:, i] = job_series.str.find(adj).replace(0, 1).replace(-1, 0).to_numpy(dtype=int)
        one_hot_j[:, i] = job_series.str.contains(adj).to_numpy(dtype=int)
    features_matrix = np.append(features_matrix, one_hot_j, axis=1)

    # degree = df.iloc[:, 8]
    degree = df["University Degree"].str.lower()
    unq_degree = degree.unique()
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

    # glasses = df.iloc[:, 9]
    glasses = df["Wears Glasses"].to_numpy(dtype=int)
    # Unique -> [0 1]
    features_matrix = np.append(features_matrix, glasses.reshape(l, 1), axis=1)

    # # hair = df.iloc[:, 10]
    # hair = df["Hair Color"].str.lower()
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

    # height = df.iloc[:, 11]
    height = df["Body Height [cm]"].to_numpy(dtype=int)
    # Unique ->
    gm_height = int(gmean(height))
    stats.update({"height_mean": gm_height})
    height = height / gm_height
    features_matrix = np.append(features_matrix, height.reshape(l, 1), axis=1)

    return features_matrix, income, stats


def process_test(df, stats):
    """
    To manipulate the training data
    :param df: pandas dataframe
    :return:
    """

    """
    1. Instance - Instance of data (irrelevant)

    2. Year of Record -  Year record was made (possibly unimportant) - scale on geo mean
    3. Gender - Participant gender (relevant) - encode -> (male) (female) (other) - "Bad" -> (other)
    4. Age - Age of participant (relevant) - scale on min/max - remove top ages, too old
    5. Country - Country participant is from/works in (relevant) - use ISO country code ? -- one hot encode
    6. Size of City - Population size (somewhat relevant) - scale from ISO ? Aggregate with country in some way ?  -- encode per 10,000 ?
    7. Profession - Participant job (relevant) - try to categorise (use salary as metric) ? - word encoding
    8. University Degree - Level of education - encode -> (no) (bachelor) (master) (phd) - "Bad" -> (no)
    9. Wears Glasses - Boolean glasses wearing (possibly unimportant, age correlated ?) - encode (already ?) -> (no) (yes) - "Bad" -> (no)
    10. Hair Colour - Hair (possible relevant) - encode ? RGB ? - "Bad" -> bald ?
    11. Body height [cm] - Tallness (relevant) - scale to meters? - over 240 highly unlikely, ignore, skim top 5% ?

    12. Income in EUR - income - leas as is - desired output - remove negative incomes
    :return:
    """
    l = len(df)
    features_matrix = np.ones([l, 1])

    # instance = df.iloc[:, 0]
    instance = df["Instance"].to_numpy(dtype=int)
    # plt.figure()
    # plt.scatter(instance, instance)
    # plt.xlabel("Instance")
    # plt.ylabel("Instance")
    # plt.show()

    # year_record = df.iloc[:, 1]
    year_record = df["Year of Record"].to_numpy(dtype=int)
    # Unique ->
    year_record = pd.Series(year_record)
    f_year_record = year_record.where(lambda x: x > 0).dropna().to_numpy(dtype=int)
    # n_year_record = year_record.where(lambda x: x <= 0).to_numpy(dtype=int)
    # max_year = max(f_year_record)
    # min_year = min(f_year_record)
    year_geo_mean = stats["year_mean"]
    year_record = year_record.where(lambda x: x > 0, year_geo_mean).to_numpy(dtype=int) / year_geo_mean
    features_matrix = np.append(features_matrix, year_record.reshape(l, 1), axis=1)
    # plt.figure()
    # plt.scatter(year_record, instance)
    # plt.xlabel("Instance")
    # plt.ylabel("Year of Record")
    # plt.show()

    # gender = df.iloc[:, 2]
    gender = df["Gender"].str.lower() #.to_numpy(dtype=str)
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

    # age = df.iloc[:, 3]
    age_df = df["Age"]
    gm_age = stats["age_mean"]
    age = age_df.where(lambda x: x > 0, gm_age).to_numpy(dtype=int) / gm_age
    features_matrix = np.append(features_matrix, age.reshape(l, 1), axis=1)

    # country = df.iloc[:, 4]
    country = df["Country"].str.lower()
    # Unique ->
    # Laos -> Lao
    # North Korea -> KOR
    # country_df = country.replace(
    #     #         "Laos", "LAO").replace(
    #     #         "South Korea", "KOR").replace(
    #     #         "North Korea", "PRK").replace(
    #     #         "DR Congo", "COD").unique()
    # country_df = country.unique()
    country_list = stats["country_list"]
    one_hot_c = np.zeros([l, len(country_list)])
    for i, c in enumerate(country_list):
        one_hot_c[:, i] = country.where(lambda x: x == c, 0).replace(c, 1).to_numpy(dtype=int)
    features_matrix = np.append(features_matrix, one_hot_c, axis=1)
    # searched_countries = pd.Series(
    #     country.replace(
    #         "Laos", "LAO").replace(
    #         "South Korea", "KOR").replace(
    #         "North Korea", "PRK").replace(
    #         "DR Congo", "COD").unique()
    # ).apply(pyc.countries.search_fuzzy).tolist()

    # population = df.iloc[:, 5]
    population = df["Size of City"]
    pop_mean = stats["pop_mean"]
    population = population.where(lambda x: x > 0).fillna(pop_mean).to_numpy(dtype=int) / pop_mean
    # max_pop = int(population.max())
    # min_pop = int(population.min())
    features_matrix = np.append(features_matrix, population.reshape(l, 1), axis=1)

    # job = df.iloc[:, 6]
    job_series = df["Profession"].str.lower().fillna("other")
    # split_job = pd.Series(job_series.unique()).apply(lambda x: x.split(" ")).tolist()
    # job_adj_set = set()
    # for job_list in split_job:
    #     for job in job_list:
    #         job_adj_set.add(job)
    # job_adj_list = sorted(list(job_adj_set))
    # pandas_categories = job_series.astype("category").cat.codes.to_numpy().reshape([l, 1])
    # features_matrix = np.append(features_matrix, pandas_categories, axis=1)
    job_adj_list = stats["job_list"]
    one_hot_j = np.zeros([l, len(job_adj_list)])
    for i, adj in enumerate(job_adj_list):
        # one_hot_j[:, i] = job_series.where(lambda x: adj in x).fillna(0)
        # one_hot_j[:, i] = job_series.str.find(adj).replace(0, 1).replace(-1, 0).to_numpy(dtype=int)
        one_hot_j[:, i] = job_series.str.contains(adj).to_numpy(dtype=int)
    features_matrix = np.append(features_matrix, one_hot_j, axis=1)

    # degree = df.iloc[:, 8]
    degree = df["University Degree"].str.lower()
    unq_degree = degree.unique()
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

    # glasses = df.iloc[:, 9]
    glasses = df["Wears Glasses"].to_numpy(dtype=int)
    # Unique -> [0 1]
    features_matrix = np.append(features_matrix, glasses.reshape(l, 1), axis=1)

    # # hair = df.iloc[:, 10]
    # hair = df["Hair Color"].str.lower()
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

    # height = df.iloc[:, 11]
    height = df["Body Height [cm]"].to_numpy(dtype=int)
    # Unique ->
    gm_height = stats["height_mean"]
    height = height / gm_height
    features_matrix = np.append(features_matrix, height.reshape(l, 1), axis=1)

    # income = df.iloc[:, 12]
    # income = df["Income in EUR"].to_numpy(dtype=float).reshape(l, 1)

    return features_matrix


def cleanup(return_dir, tmp_dir):
    cwd = os.getcwd()
    os.chdir(return_dir)
    shutil.rmtree(cwd)
    if os.path.exists(cwd):
        raise FileExistsError("{} exists".format(tmp_dir))
    os.chdir(return_dir)


def write_predictions(df, predictions, output_file, data_dir, tmp_dir):
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


def run(test_size, training=True):
    print("\nTest size: {}\n".format(test_size))
    script_dir = os.getcwd()
    root_dir = os.path.dirname(script_dir)
    os.chdir(root_dir)
    data_dir = os.path.join(root_dir, "Data")
    # if os.path.exists(data_dir):
    #     print(data_dir)
    # tmp_dir = tempfile.mkdtemp(dir=root_dir)

    tmp_dir = os.path.join(root_dir, "tmp")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    os.chdir(data_dir)
    training_file = shutil.copy(FILES.get("training"), tmp_dir)
    os.chdir(tmp_dir)
    training_data = get_data(training_file)
    x, y, stats = process_training(training_data)
    # poly = PolynomialFeatures(degree=2)
    # X_ = poly.fit_transform(x)
    # re_model = linear_model.LinearRegression()
    re_model = linear_model.Ridge(alpha=0.1, normalize=False, fit_intercept=False)
    # re_model = linear_model.Lasso()
    # re_model = linear_model.RidgeCV(fit_intercept=False)

    if training:
        x_train, x_val, y_train, y_val = model_selection.train_test_split(x, y, test_size=0.2, random_state=1)

        # Use training data
        re_model.fit(x_train, y_train)

        # Use validation data
        y_val_pred = re_model.predict(x_val)

        pred_df = pd.DataFrame({"Val": y_val.flatten(),
                                "Pred": y_val_pred.flatten(),
                                "Diff": abs(y_val - y_val_pred).flatten()})

        # Prediction stats
        print(pd.DataFrame(y_val_pred).describe())
        # The coefficients
        print('Coefficients: \n', re_model.coef_)
        # The mean squared error
        mse = sk.metrics.mean_squared_error(y_val, y_val_pred)
        rmse = sqrt(mse)
        print("Mean squared error: {:,.2f}".format(mse))
        print("RMSE: {:,.5f}".format(rmse))
        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' % sk.metrics.r2_score(y_val, y_val_pred))

        # Plot outputs
        # plt.figure()
        # plt.scatter(x_val[:, 4], y_val, color='black')
        # plt.plot(x_val[:, 4], y_val_pred, color='blue', linewidth=3)
        #
        # plt.xticks(())
        # plt.yticks(())
        #
        # plt.show()

    else:
        # Output test predictions
        os.chdir(data_dir)
        test_file = shutil.copy(FILES.get("test"), tmp_dir)
        os.chdir(tmp_dir)

        # Use training data
        re_model.fit(x, y)

        test_data = get_data(test_file)
        x_test = process_test(test_data, stats)
        y_test_pred = re_model.predict(x_test)

        write_predictions(test_data, y_test_pred, test_file, data_dir, tmp_dir)

    os.chdir(script_dir)
    # cleanup(script_dir, tmp_dir)


def feature_correlations(test_df, correlation_feature="Income in EUR"):
    df = test_df.copy()
    columns = list(df.columns.values)
    for column in columns:
        if column != correlation_feature:
            df[column] = df[column].astype("category").cat.codes

    correlations = df[df.columns[1:]].corr()[correlation_feature][:]
    print(correlations)

    return correlations


def plot_relations(x, y, x_label, y_label="Income"):
    plt.figure()
    x, y = list(zip(*sorted(list(zip(x, y.flatten())), key=lambda x: x[0])))
    plt.scatter(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


if __name__ == '__main__':
    # test_sizes = np.arange(0, 0.5, 0.005)
    # for test_size in test_sizes:
    #     run(test_size, training=True)

    run(0.1, training=True)

    # script_dir = os.getcwd()
    # root_dir = os.path.dirname(script_dir)
    # os.chdir(root_dir)
    # data_dir = os.path.join(root_dir, "Data")
    # if os.path.exists(data_dir):
    #     print(data_dir)
    # # tmp_dir = tempfile.mkdtemp(dir=root_dir)
    #
    # tmp_dir = os.path.join(root_dir, "tmp")
    # if os.path.exists(tmp_dir):
    #     shutil.rmtree(tmp_dir)
    # os.makedirs(tmp_dir)
    #
    # os.chdir(data_dir)
    # training_file = shutil.copy(FILES.get("training"), tmp_dir)
    # os.chdir(tmp_dir)
    # training_data = get_data(training_file)
    # x, y, stats = process_training(training_data, remove_negative=True, use_ones=False)
    #
    # x_train, x_val, y_train, y_val = model_selection.train_test_split(x, y, test_size=0.3, random_state=1)
    #
    # # Use training data
    # # re_model = linear_model.LinearRegression()
    # re_model = linear_model.Ridge(alpha=1.0, normalize=True)
    # re_model.fit(x_train, y_train)
    #
    # # Use validation data
    # y_val_pred = re_model.predict(x_val)
    #
    # # The coefficients
    # print('Coefficients: \n', re_model.coef_)
    # # The mean squared error
    # mse = sk.metrics.mean_squared_error(y_val, y_val_pred)
    # rmse = sqrt(mse)
    # print("Mean squared error: {:,.2f}".format(mse))
    # print("RMSE: {:,.5f}".format(rmse))
    # # Explained variance score: 1 is perfect prediction
    # print('Variance score: %.2f' % sk.metrics.r2_score(y_val, y_val_pred))
    #
    # # Plot outputs
    # # plt.figure()
    # # plt.scatter(x_val[:, 4], y_val, color='black')
    # # plt.plot(x_val[:, 4], y_val_pred, color='blue', linewidth=3)
    # #
    # # plt.xticks(())
    # # plt.yticks(())
    # #
    # # plt.show()
    #
    # # Output test predictions
    # # os.chdir(data_dir)
    # # test_file = shutil.copy(FILES.get("test"), tmp_dir)
    # # os.chdir(tmp_dir)
    # # test_data = get_data(test_file)
    # # x_test = process_test(test_data, stats, use_ones=True)
    # # y_test_pred = re_model.predict(x_test)
    # #
    # # # Write to test file
    # # test_data["Income"] = y_test_pred
    # # test_data.to_csv(test_file, index=False)
    # # # Write to submission file
    # # os.chdir(data_dir)
    # # submission_file = shutil.copy(FILES["submission"]["use"], tmp_dir)
    # # os.chdir(tmp_dir)
    # # submission_df = get_data(submission_file)
    # # submission_df["Income"] = y_test_pred
    # # submission_df.to_csv(submission_file, index=False)
    #
    # os.chdir(script_dir)
    # # cleanup(script_dir)