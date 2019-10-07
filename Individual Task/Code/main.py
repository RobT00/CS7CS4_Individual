import os
import tempfile
import shutil
import pandas as pd
import numpy as np
import sklearn as sk
import scipy as sp
from scipy.stats.mstats import gmean
import matplotlib.pyplot as plt
import pycountry as pyc

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


def process(df):
    """
    To manipulate the training data
    :param df: pandas dataframe
    :return:
    """

    """
    1. Instance - Instance of data (irrelevant)

    2. Year of Record -  Year record was made (possibly unimportant) - scale on geo mean
    3. Gender - Participant gender (relevant) - encode -> (male) (female) (other) - "Bad" -> (other)
    4. Age - Age of participant (relevant) - scale on min/max
    5. Country - Country participant is from/works in (relevant) - use ISO country code ? -- one hot encode
    6. Size of City - Population size (somewhat relevant) - scale from ISO ? Aggregate with country in some way ?  -- encode per 10,000 ?
    7. Profession - Participant job (relevant) - try to categorise (use salary as metric) ?
    8. University Degree - Level of education - encode -> (no) (bachelor) (master) (phd) - "Bad" -> (no)
    9. Wears Glasses - Boolean glasses wearing (possibly unimportant, age correlated ?) - encode (already ?) -> (no) (yes) - "Bad" -> (no)
    10. Hair Colour - Hair (possible relevant) - encode ? RGB ? - "Bad" -> bald ?
    11. Body height [cm] - Tallness (relevant) - scale to meters?

    12. Income in EUR - income - leas as is - desired output
    :return:
    """
    l = len(df)
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
    year_record = year_record.where(lambda x: x > 0).fillna(year_geo_mean).to_numpy(dtype=int) / year_geo_mean
    # plt.figure()
    # plt.scatter(year_record, instance)
    # plt.xlabel("Instance")
    # plt.ylabel("Year of Record")
    # plt.show()
    # gender = df.iloc[:, 2]
    gender = df["Gender"] #.to_numpy(dtype=str)
    # Unique -> ['0' 'other' 'female' 'male' nan 'unknown']
    male = gender.where(lambda x: x.str.lower() == "male").fillna(0).replace("male", 1).to_numpy(dtype=int)
    female = gender.where(lambda x: x.str.lower() == "female").fillna(0).replace("female", 1).to_numpy(dtype=int)
    other = pd.Series(male + female).replace(1, 2).replace(0, 1).replace(2, 0).to_numpy(dtype=int)
    # np.append(male.reshape(111993, 1), np.append(female.reshape(111993, 1), other.reshape(111993, 1), axis=1), axis=1)
    # age = df.iloc[:, 3]
    age_df = df["Age"]
    gm_age = int(gmean(age_df.where(lambda x: x > 0).dropna().to_numpy(dtype=int)))
    age = age_df.where(lambda x: x > 0).fillna(gm_age).to_numpy(dtype=int) / gm_age
    # country = df.iloc[:, 4]
    country = df["Country"]
    # Unique ->
    country_df = country.replace(
            "Laos", "LAO").replace(
            "South Korea", "KOR").replace(
            "North Korea", "PRK").replace(
            "DR Congo", "COD").str.lower().unique()
    country_list = sorted(country_df.tolist())
    one_hot_c = np.zeros([l, len(country_df)])
    for i, c in enumerate(country_list):
        one_hot_c[i, :] = country.where(lambda x: x.str.lower() == c).fillna(0).replace(c, 1).to_numpy(dtype=str)
    searched_countries = pd.Series(
        country.replace(
            "Laos", "LAO").replace(
            "South Korea", "KOR").replace(
            "North Korea", "PRK").replace(
            "DR Congo", "COD").unique()
    ).apply(pyc.countries.search_fuzzy).tolist()
    # population = df.iloc[:, 5]
    population = df["Size of City"]
    # Laos -> Lao
    # North Korea -> KOR
    # job = df.iloc[:, 6]
    job = df["Profession"]
    # Unique ->
    # degree = df.iloc[:, 8]
    degree = df["University Degree"]
    # Unique ->
    # glasses = df.iloc[:, 9]
    glasses = df["Wears Glasses"]
    # Unique ->
    # hair = df.iloc[:, 10]
    hair = df["Hair Color"]
    # Unique ->
    # height = df.iloc[:, 11]
    height = df["Body Height [cm]"]
    # Unique ->

    # income = df.iloc[:, 12]
    income = df["Income in EUR"]

    return


def cleanup(return_dir):
    cwd = os.getcwd()
    os.chdir(return_dir)
    shutil.rmtree(cwd)
    if os.path.exists(cwd):
        raise FileExistsError("{} exists".format(tmp_dir))
    os.chdir(return_dir)


if __name__ == '__main__':
    script_dir = os.getcwd()
    root_dir = os.path.dirname(script_dir)
    os.chdir(root_dir)
    data_dir = os.path.join(root_dir, "Data")
    if os.path.exists(data_dir):
        print(data_dir)
    # tmp_dir = tempfile.mkdtemp(dir=root_dir)

    tmp_dir = os.path.join(root_dir, "tmp")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    os.chdir(data_dir)
    training_file = shutil.copy(FILES.get("training"), tmp_dir)
    os.chdir(tmp_dir)
    training_data = get_data(training_file)
    process(training_data)

    # cleanup(script_dir)