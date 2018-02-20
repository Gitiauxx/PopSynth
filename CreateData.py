from utilities import categorize, mapping_household_marg_to_sample, mapping_person_marg_to_sample
from censusHelper import Census
import re
import pandas as pd


class Dataset(object):
    def __init__(self, base_url, key, pums_url, geo_url, year, variables, variables_p=None):

        self.variables = variables
        self.variables_p = variables_p
        self.c = Census(key, base_url, year, pums_url, geo_url=geo_url)
        self._CACHE = {}

    def _to_columns(self, variables):

        """
        Take a dictionary of tuples (cat, ACS cols)
        :return: all ACS columns to be extraced
        """

        col_list = []
        for _, expr in variables.items():
            col_list = col_list + re.findall(r"[\w]+", expr)

        return col_list

    @property
    def h_marginal_acs(self):
        """
        This function transform the fields queried from the acs into
        categories
        :return: table with nrows=number of block group and ncolumns = number of category * number of sub caterogies
         within each category
        """
        block_group_columns = self._to_columns(self.variables)
        df = self.c.query_block_group(block_group_columns)
        df['GEOID'] = df['state'] + df['county'] + df['tract'] + df['block group']
        cat_df = categorize(df, self.variables, index_cols=['GEOID'])

        if self.variables_p is not None:
            block_group_columns = self._to_columns(
                self.variables_p)  # query acs persons level characteristics if necessary
            df_p = self.c.query_block_group(block_group_columns)
            df_p['GEOID'] = df_p['state'] + df_p['county'] + df_p['tract'] + df_p['block group']
            cat_p = categorize(df_p, self.variables_p, index_cols=['GEOID'])
            cat_df = cat_df.join(cat_p)

        return cat_df

    @property
    def pums_h(self):
        """
        :return: households sample from the pums
        """
        if 'pums_h' not in self._CACHE:
            pums_h = self.c.get_pums_h().set_index('SERIALNO')
            self._CACHE['pums_h'] = pums_h

        else:
            pums_h = self._CACHE['pums_h']

        return pums_h[pums_h.WGTP > 0]

    @property
    def pums_p(self):
        """
        :return: person sample from the pums
        """
        if 'pums_p' not in self._CACHE:
            pums_p = self.c.get_pums_p().set_index(['SERIALNO', 'SPORDER'])
            self._CACHE['pums_p'] = pums_p
        else:
            pums_p = self._CACHE['pums_p']
        return pums_p

    @property
    def sample_df(self):

        """
        This function converts the households sample into a table with category variables
        The columns have the same name as the ones obtained from the ACS marginals
        Variables are equal to one if and only if they belong to the category defined in the ACS table
        The function also adds the characteristics from the individuals living in each household
        Indivudual-level characteristics are properly adjusted to account for the ratio of individual weight
        within the household
        :return:
        """
        sample_df = pd.DataFrame(index=self.pums_h.index)
        cat = list(set(index[0] for index, _ in self.variables.items()))

        for index in cat:  # convert sample variables into category using mapping_household_arg_to_sample

            sample_df[index] = self.pums_h.apply(mapping_household_marg_to_sample()[index], axis=1)
            sample_df = sample_df[sample_df[index] != '-99']

        data = pd.get_dummies(sample_df)  # convert category into dummies

        if self.variables_p is not None:  # add individual-level variables if any is included in the model
            sample_p = self.sample_p
            for v in list(sample_p.columns):
                sample_p[v] = sample_p[v] / self.weight  # reweighting of individual-level variables
            data = data.join(sample_p)
        return data

    @property
    def weight(self):
        """

        :return: housihold sampling weight
        """
        if 'weight' not in self._CACHE:
            w = self.pums_h.WGTP
            self._CACHE['weigt'] = w
        else:
            w = self._CACHE['weigt']
        return w

    @property
    def sample_p(self):

        """
        This function converts the persons sample into a table with category variables
        aggregated at the household level using persons weight
        The columns have the same name as the ones obtained from the ACS marginals (persons variables)
        :return:
        """
        sample = pd.DataFrame(index=self.pums_p.index)
        cat = list(set(index[0] for index, _ in self.variables_p.items()))

        for index in cat:  # create category from sample data
            sample[index] = self.pums_p.apply(mapping_person_marg_to_sample()[index], axis=1)
            sample = sample[sample[index] != '-99']

        df = pd.get_dummies(sample)  # transform category into dummies
        W = self.pums_p['PWGTP']

        for v in df.columns:
            df[v] = df[v] * W  # add individual weights

        df = df.reset_index().drop('SPORDER', axis=1)
        return df.groupby('SERIALNO').sum()  # aggregate at the households levels


if __name__ == "__main__":
    import numpy as np
    import os

    key = "9d119de5f3de42bf4570723644941f4a4a707b8f"
    base_url = 'https://api.census.gov/data/year/acs/acs5?get=NAME,variables&for=block%20group:*&in=in_field&key=your_key'
    geo_url = 'https://www2.census.gov/geo/docs/reference/codes/files/st08_co_cou.txt'
    pums_url = 'https://www2.census.gov/programs-surveys/acs/data/pums/year/5-Year/'

    eval_d = {

        ("hhincome", "lt30"):
            "B19001_002E + B19001_003E + B19001_004E + "
            "B19001_005E + B19001_006E",
        ("hhincome", "gt30-lt60"):
            "B19001_007E + B19001_008E + B19001_009E + "
            "B19001_010E + B19001_011E",
        ("hhincome", "gt60-lt100"): "B19001_012E + B19001_013E",
        ("hhincome", "gt100-lt150"): "B19001_014E + B19001_015E",
        ("hhincome", "gt150"): "B19001_016E + B19001_017E",

        ("nhouseholds", "all"): "B19001_001E",

        ("tenure", "own"): "B25038_002E",
        ("tenure", "rent"): "B25038_009E",

        ("seniors", "yes"): "B11007_002E",
        ("seniors", "no"): "B11007_007E",

        ("children", "yes"): "B11005_002E",
        ("children", "no"): "B11005_011E",

        ("persons", "1 persons"): "B11016_010E",
        ("persons", "2 persons"): "B11016_003E + B11016_011E",
        ("persons", "3 persons"): "B11016_004E + B11016_012E",
        ("persons", "4 persons"): "B11016_005E + B11016_013E",
        ("persons",
         "5 persons or more"): "B11016_006E + B11016_014E + B11016_008E + B11016_016E + B11016_007E + B11016_015E"

    }

    eval_p = {

        ("sex", "male"): "B01001_002E * B11002_001E /B01001_001E",
        ("sex", "female"): "B01001_026E * B11002_001E /B01001_001E",

        ("age", "19 and under"): "(B01001_003E + B01001_004E + B01001_005E + "
                                 "B01001_006E + B01001_007E + B01001_027E + "
                                 "B01001_028E + B01001_029E + B01001_030E + "
                                 "B01001_031E)* B11002_001E / B01001_001E",
        ("age", "20 to 34"): "(B01001_008E + B01001_009E + B01001_010E + "
                             "B01001_011E + B01001_012E + B01001_032E + "
                             "B01001_033E + B01001_034E + B01001_035E + "
                             "B01001_036E)* B11002_001E / B01001_001E",
        ("age", "35 to 59"): "(B01001_013E + B01001_014E + B01001_015E + "
                             "B01001_016E + B01001_017E + B01001_037E + "
                             "B01001_038E + B01001_039E + B01001_040E + "
                             "B01001_041E)* B11002_001E / B01001_001E",
        ("age", "60 and above"): "(B01001_018E + B01001_019E + B01001_020E + "
                                 "B01001_021E + B01001_022E + B01001_023E + "
                                 "B01001_024E + B01001_025E + B01001_042E + "
                                 "B01001_043E + B01001_044E + B01001_045E + "
                                 "B01001_046E + B01001_047E + B01001_048E + "
                                 "B01001_049E)* B11002_001E /B01001_001E",

        ("grade", "PK"): "B14007_003E * B11002_001E / B01001_001E",
        ("grade", "K"): "B14007_004E * B11002_001E /B01001_001E",
        ("grade", "G1_4"): "(B14007_005E + B14007_006E + B14007_007E + B14007_008E)* B11002_001E / B01001_001E",
        ("grade", "G5_8"): "(B14007_009E + B14007_010E + B14007_011E + B14007_012E)* B11002_001E / B01001_001E",
        ("grade", "G9_12"): "(B14007_013E + B14007_014E + B14007_015E + B14007_016E)* B11002_001E / B01001_001E",
        ("grade", "college_under"): "B14007_017E * B11002_001E / B01001_001E",
        ("grade", "graduate"): "B14007_018E * B11002_001E / B01001_001E",
        ("grade", "not enrolled"): "(B14007_019E + B01001_001E - B14007_001E) * B11002_001E / B01001_001E"

    }

    dataset = Dataset(base_url, key, pums_url, geo_url, 2016, eval_d, variables_p=eval_p)
    W = dataset.weight
    M = dataset.h_marginal_acs
    X = dataset.sample_df
    W = W.loc[X.index]

    var = list(M.drop('nhouseholds_all', axis=1).columns) + ['nhouseholds_all']
    X = X[var]
    M = M[var]

    # export csv files
    input_directory = 'C:\\Users\\xgitiaux\\Documents\\Research and Analysis\\Synthetizer\\Data\\Inputs'
    filename_x = os.path.join(input_directory, 'sample.csv')
    filename_m = os.path.join(input_directory, 'marginal.csv')
    filename_w = os.path.join(input_directory, 'weight.csv')
    X.to_csv(filename_x)
    M.to_csv(filename_m)
    W.to_csv(filename_w)

    # export numpy array
    M = np.array(M)
    X = np.array(X)

    filename_x = os.path.join(input_directory, 'sample.npy')
    filename_m = os.path.join(input_directory, 'marginal.npy')
    filename_w = os.path.join(input_directory, 'weight.npy')

    np.save(filename_x, X)
    np.save(filename_w, W)
    np.save(filename_m, M)
