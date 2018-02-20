import pandas as pd
import yaml


def categorize(df, eval_d, index_cols=None):
    """

    :param df: data
    :param eval_d: dictionary with expression to evaluate
    :param index_cols: index
    :return: df with expression evaluated and index set to index_col
    """
    cat_df = pd.DataFrame(index=df.index)

    for index, expr in eval_d.items():
        cat_df[index[0] + '_' + index[1]] = df.eval(expr)

    if index_cols is not None:
        cat_df[index_cols] = df[index_cols]
        cat_df = cat_df.set_index(index_cols)

    cat_df = cat_df.sort_index(axis=1)

    return cat_df


def mapping_household_marg_to_sample():
    def cars_cat(r):
        if r.VEH == 0:
            return "none"
        elif r.VEH == 1:
            return "one"
        elif r.VEH in [2, 3, 4, 5, 6]:
            return "two or more"
        else:
            return '-99'

    def children_cat(r):
        if (r.HUPAC > 0) & (r.HUPAC < 4):
            return "y"
        elif r.HUPAC == 4:
            return "n"
        else:
            return '-99'

    def age_of_head_cat(r):
        if r.age_of_head < 35:
            return "lt35"
        elif r.age_of_head >= 65:
            return "gt65"
        return "gt35-lt65"

    def workers_cat(r):
        if r.WIF == 3:
            return "two or more"
        elif r.WIF == 2:
            return "two or more"
        elif r.WIF == 1:
            return "one"
        else:
            return "none"

    def income_cat(r):
        if r.HINCP < 30000:
            return "lt30"
        elif r.HINCP < 60000:
            return "gt30-lt60"
        elif r.HINCP < 100000:
            return "gt60-lt100"
        elif r.HINCP < 150000:
            return "gt100-lt150"
        elif r.HINCP >= 150000:
            return "gt150"
        else:
            return '-99'

    def persons_cat(r):
        if r.NP <= 1:
            return "1 person"
        elif r.NP <= 2:
            return "2 persons"
        elif r.NP <= 3:
            return "3 persons"
        elif r.NP <= 4:
            return "4 persons"
        elif r.NP > 4:
            return "5 persons or more"
        else:
            return '-99'

    def payment_cat(r):
        if r.TEN <= 2:
            if r.VALP < 100000:
                return "own01"
            elif 100000 <= r.VALP < 300000:
                return "own02"
            elif 300000 <= r.VALP < 500000:
                return "own03"
            elif r.VALP >= 500000:
                return "own04"

        elif r.TEN == 3:
            if r.GRNTP < 500:
                return "rent01"
            elif 500 <= r.GRNTP < 1000:
                return "rent02"
            elif 1000 <= r.GRNTP:
                return "rent03"
        elif r.TEN == 4:
            return "norent"

    def seniors_cat(r):
        if r.R65 > 0:
            return "yes"
        elif r.R65 == 0:
            return "no"
        else:
            return '-99'

    def tenure(r):
        if (r.TEN < 3) & (r.TEN >= 0):
            return "own"
        elif (r.TEN >= 3):
            return "rent"
        else:
            return '-99'

    def number_household_cat(r):
        return "all"

    mappingfunction = {"hhincome": income_cat, "persons": persons_cat, "nhouseholds": number_household_cat,
                       "payment": payment_cat, "tenure": tenure, 'cars': cars_cat,
                       "persons": persons_cat, "children": children_cat, "seniors": seniors_cat}

    return mappingfunction


def mapping_person_marg_to_sample():
    def age_cat(r):

        if r.AGEP <= 19:
            return "19 and under"
        elif r.AGEP < 35:
            return "20 to 34"
        elif r.AGEP < 60:
            return "35 to 59"
        elif r.AGEP >= 60:
            return "60 and above"
        else:
            return '-99'

    def race_cat(r):
        if r.RAC1P == 1:
            return "white"
        elif r.RAC1P == 2:
            return "black"
        elif r.RAC1P == 6:
            return "asian"
        elif r.RAC1P in [3, 4, 5, 7, 8, 9]:
            return "other"

    def sex_cat(r):
        if r.SEX == 1:
            return "male"
        elif r.SEX == 2:
            return "female"
        else:
            return '-99'

    def commute_cat(r):
        if r.JWTR == 1:
            if r.JWRIP == 1:
                return "drive alone"
            elif r.JWRIP in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
                return "carpool"
        elif r.JWTR in [2, 3, 4, 5, 6]:
            return "transit"
        elif r.JWTR in [7, 8, 12]:
            return "others"
        elif r.JWTR in [9, 10]:
            return "bike or walk"
        elif r.JWTR == 11:
            return "work at home"
        else:
            return "less than 16"

    def emp_cat(r):
        if r.ESR in [1, 2, 3, 4, 5]:
            return "in labor force"
        elif r.ESR == 6:
            return "not in labor force"
        else:
            return "less than 16"

    def grade_cat(r):
        if r.SCHG == 1:
            return "PK"
        elif r.SCHG == 2:
            return "K"
        elif r.SCHG in [3, 4, 5, 6]:
            return "G1_4"
        elif r.SCHG in ([7, 8, 9, 10]):
            return "G5_8"
        elif r.SCHG in [11, 12, 13, 14]:
            return "G9_12"
        elif r.SCHG == 15:
            return "college_under"
        elif r.SCHG == 16:
            return "graduate"
        else:
            return "not enrolled"

    return {"age": age_cat, "race": race_cat, "sex": sex_cat,
            "emp": emp_cat, "grade": grade_cat,
            "commute": commute_cat
            }


def from_yaml_to_ACS(var_list, yaml_str=None, str_or_buffer=None):
    """

    :param var_list: list of variables/cat to include in the joint distribution estimation
    :param yaml_str: yaml-type string
    :param str_or_buffer: yaml filename
    :return: a dictionary of the form {(cat, subcat): operation on ACS columns}
    """

    if not yaml_str and not str_or_buffer:
        raise ValueError('One of yaml_str or str_or_buffer is required.')

    if yaml_str:
        cfg = yaml.load(yaml_str)
    elif isinstance(str_or_buffer, str):
        with open(str_or_buffer) as f:
            cfg = yaml.load(f)
    else:
        cfg = yaml.load(str_or_buffer)

    if var_list is None:  # handle the case where there is no persons or households characteristics
        return None

    dict_var = {}
    for var in var_list:  # loop over all the categories to include in the estimation

        if var not in cfg:  # check that var is in the config file
            raise ValueError('The variable is not defined in the config file')

        cats = cfg[var]
        for subcat in cats:
            dict_var[(var, subcat)] = cats[subcat]

    return dict_var


if __name__ == '__main__':
    yamlfile = '..\\Data\Inputs\config_ACS_variables.yaml'
    print(from_yaml_to_ACS(['nhouseholds', 'hhincome', 'tenure', 'persons', 'age'], str_or_buffer=yamlfile))
