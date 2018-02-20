from Solvers import softmax_descent
from utilities import from_yaml_to_ACS
import yaml
from CreateData import Dataset
import numpy as np
import time
import pandas as pd


class synthetize(object):
    def __init__(self, year, variable_h, census_config, variables_p=None):
        self.year = year
        self.__variable_h = variable_h
        self.census_config = census_config
        self.__variable_p = variables_p
        self._CACHE = {}

    @classmethod
    def from_config(cls, yaml_str=None, str_or_buffer=None):

        """
        create an input configuration from a saved yaml file
        parameters
        ----------
        yaml_str : str, optional
            A YAML string from which to load model.
        str_or_buffer : str or file like, optional
            File name or buffer from which to load YAML.

        :return:
        a synthetizer model
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

        variables_h = cfg['Households Variables']
        year = cfg['year']
        cenus_config = cfg['census configuration']

        if 'Persons variables' in cfg:
            variables_p = cfg['Persons variables']
        else:
            variables_p = None

        synth = cls(year,
                    variables_h,
                    cenus_config,
                    variables_p=variables_p)  # create a synthetize method

        return synth

    @property
    def variables(self):
        acs_config = self.census_config['config acs']
        return from_yaml_to_ACS(self.__variable_h, str_or_buffer=acs_config)

    @property
    def variables_p(self):
        acs_config = self.census_config['config acs']
        return from_yaml_to_ACS(self.__variable_p, str_or_buffer=acs_config)

    @property
    def __dataset(self):

        if 'datasetFunction' not in self._CACHE:

            census_config = self.census_config
            key = census_config['key']
            pums_url = census_config['pums_url']
            base_url = census_config['base_url']
            geo_url = census_config['geo_url']

            dt = Dataset(base_url, key, pums_url, geo_url, self.year, self.variables, self.variables_p)
            self._CACHE['datasetFunction'] = dt

        else:
            dt = self._CACHE['datasetFunction']

        return dt

    @property
    def marginal_acs(self):

        if 'marginals' not in self._CACHE:
            print("Constructing marginals from the Census")
            marginals = self.__dataset.h_marginal_acs
            self._CACHE['marginals'] = marginals

        else:
            marginals = self._CACHE['marginals']

        return marginals

    @property
    def sample(self):

        if 'sample' not in self._CACHE:
            print("Constructing sample datasets from the Census")
            sample = self.__dataset.sample_df
            self._CACHE['sample'] = sample
        else:
            sample = self._CACHE['sample']

        return sample

    @property
    def weight(self):
        if 'weight' not in self._CACHE:
            print("Constructing weight dataset from the Census")
            weight = self.__dataset.weight
            self._CACHE['weight'] = weight
        else:
            weight = self._CACHE['weight']

        return weight

    def to_matrix(self):

        dfm = self.marginal_acs
        col = list(dfm.drop('nhouseholds_all', axis=1).columns)  # adjust columns to make sure they are all consistent
        col = col + ['nhouseholds_all']  # the last column is the count of households by geography
        dfx = self.sample[col]
        dfm = dfm[col][dfm.nhouseholds_all > 0]
        dfw = self.weight.loc[dfx.index]  # keep only the households that remains in the sample

        X = np.array(dfx)  # pums sample as  #househols * #characteristics matrix
        M = np.array(dfm)  # marginal as a #geographies * #characteristics matrix
        W = np.array(dfw)  # weight as a #households vector

        XSUM = np.dot(np.transpose(X), W)  # sum for each characterisitic from the expanded sample
        MSUM = np.sum(M, axis=0)  # sum for each characterisitic from the ACS marginals
        DIFF_M_X = (XSUM - MSUM) / MSUM * 100  # percentage difference between XSUM and MSUM

        if (np.abs(DIFF_M_X) > 5).any():
            raise ValueError('The weighted sample does not appear to match the total ACS marginals'
                             'Look at the ACS config file and the Factfinder and check whether the '
                             'categories and subcategories are properly defined')

        return X, M, W

    def estimate_distribution(self):

        X, M, W = self.to_matrix()  # create input matrix and check whether they are properly balanced

        sf = softmax_descent(X, M, W, 1.0, batchSize=256)  # solver
        s = time.time()
        print("Estimating the joint distirbution of households characteristics in each location")
        _, loss = sf.epochIter()
        print("time elapsed: {:.2f}s".format(time.time() - s))

        return sf.beta, loss

    def draw(self, beta, nhouseholds):

        X, M, W = self.to_matrix()  # create input matrix and check whether they are properly balanced
        sample = self.sample
        marginal = self.marginal_acs

        SX = np.exp(np.dot(X, np.transpose(beta)))
        SXSUM = np.sum(SX, axis=1)
        SX = SX / SXSUM[:, np.newaxis]  # estimated probabilities conditional on location
        SX = np.multiply(SX, W[:, np.newaxis]) / W.sum()  # joint distribution

        NX = np.sum(SX, axis=0)  # estimated probabilities for each location

        SX = np.divide(SX, NX[np.newaxis, :])  # probability to draw a household given the location

        dh_list = []
        print("Drawwing %d households from the sample according to "
              "the estimated joint distribution" % nhouseholds)
        for i in np.arange(M.shape[0]):  # start drawing households in each location
            P = SX[:, i]
            N = int(NX[i] * nhouseholds)
            H = np.random.choice(sample.index, N, p=P, replace=True)
            dh = sample.loc[H]
            dh['GEOID'] = marginal.index[i]
            dh['GEOID']
            dh_list.append(dh[['GEOID']])

        data = pd.concat(dh_list)  # create data with index=SERIALNO and one column for geo id

        return data


if __name__ == '__main__':
    yamlfile = '..\\Data\Inputs\config_synthetizer.yaml'
    sy = synthetize.from_config(str_or_buffer=yamlfile)

    beta, loss = sy.estimate_distribution()
    print(loss)
    print(sy.draw(beta, sy.marginal_acs['nhouseholds_all'].sum()))
