import requests
import numpy as np
import pandas as pd
import json
import zipfile
from io import BytesIO


class Census(object):
    def __init__(self, key, base_url, year, pums_url, geo_url=None):
        self.key = key
        self.year = year
        self.__base_url = base_url
        self.geo_url = geo_url
        self.__pums_url = pums_url

    def to_county(self):
        return pd.read_csv(self.geo_url, header=None, usecols=[2, 3],
                           names=['county_fips', 'county_name'], dtype=str)

    def to_colnames(self, c_col):

        """
        :param c_col: a list of variables
        :return: a string of variables separated by commas
        """
        return ','.join(map(str, c_col))

    def chunks(self, l, n):
        """ Yield successive n-sized chunks from l.
        """
        for i in np.arange(0, len(l), n):
            yield l[i:i + n]

    @property
    def base_url(self):
        """
        :return: the base url with the right year in it
        """
        url = self.__base_url.replace('year', str(self.year))
        url = url.replace('your_key', self.key)
        return url

    @property
    def pums_url(self):
        """
        :return: the pum url with that right year in it
        """
        url = self.__pums_url.replace('year', str(self.year))
        url = url.replace('your_key', self.key)
        return url

    def _query(self, columns, url):

        """

        :param columns: census variables to extract
        :return: pandas table where columns are census variable names
        """
        url = url.replace('variables', self.to_colnames(columns))
        txt_extracted = requests.get(url).text
        data = json.loads(txt_extracted)

        headers = data[0]
        d = pd.DataFrame([dict(zip(headers, d)) for d in data[1:]])
        d[columns] = d[columns].astype('int32')

        return d

    def _query_in(self, columns, county):

        """

        :param columns: variables to extract
        :param county: the census api accepts only block group by county
        :return:
        """

        url = self.base_url.replace('in_field', 'state:08%20county:' + county)
        chunk_col = self.chunks(columns, 45)
        dfs = []
        for c_col in chunk_col:
            dfs.append(self._query(c_col, url))

        d = dfs[0]
        if len(dfs) > 1:
            for df in dfs[1:]:
                d = pd.merge(d, df, on="NAME", suffixes=("", "_ignore"))
                drop_cols = filter(lambda x: "_ignore" in x, d.columns)
                d = d.drop(drop_cols, axis=1)

        return d

    def query_block_group(self, columns):

        dfs_list = []

        for county in self.to_county().county_fips:
            dfs_list.append(self._query_in(columns, county))

        return pd.concat(dfs_list)

    def get_pums_h(self):
        zf = zipfile.ZipFile(BytesIO(requests.get(self.pums_url + 'csv_hco.zip').content))
        year_2digits = self.year - 2000
        pums_h = pd.read_csv(zf.open('ss' + str(year_2digits) + 'hco.csv'))
        return pums_h

    def get_pums_p(self):
        zf = zipfile.ZipFile(BytesIO(requests.get(self.pums_url + 'csv_pco.zip').content))
        year_2digits = self.year - 2000
        pums_p = pd.read_csv(zf.open('ss' + str(year_2digits) + 'pco.csv'))
        return pums_p


if __name__ == '__main__':
    key = "9d119de5f3de42bf4570723644941f4a4a707b8f"
    base_url = 'https://api.census.gov/data/year/acs/acs5?get=NAME,variables&for=block%20group:*&in=in_field&key=your_key'
    geo_url = 'https://www2.census.gov/geo/docs/reference/codes/files/st08_co_cou.txt'
    c = Census(key, base_url, 2016, geo_url=geo_url)

    income_columns = ['B19001_0%02dE' % i for i in range(1, 18)]
    age_of_head_columns = ['B25007_0%02dE' % i for i in range(1, 22)]
    tenure_mover_columns = ['B25038_0%02dE' % i for i in range(1, 16)]
    families_columns = ['B11001_001E', 'B11001_002E']

    block_group_columns = income_columns + age_of_head_columns + tenure_mover_columns + \
                          families_columns

    chunk_col = c.chunks(block_group_columns, 45)

    import time

    s = time.time()

    d = c.query_block_group(block_group_columns)
    print("time elapsed: {:.2f}s".format(time.time() - s))
    print(d)
