# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 11:34:44 2019

@author: vnandanw
"""

import pandas as pd

pd.__version__

city_name = pd.Series(['Pune','Mumbai','Nasik', 'Kolhapur','Sangli'])
populations = pd.Series([1200000,3456781,1023454,534234,450123]);

cities = pd.DataFrame({'City_name':city_name, 'Population':populations})

cities.Population = cities.Population / 1000
cities.Population
populations

#cities.Population.apply(lambda Val: cities.Population / 1000)

cities

cities['Area'] = pd.Series([100,400,47,200,45])
cities['Density'] = cities['Population'] / cities['Area']
cities

cities['Is Wide'] = ((cities['Area'] > 50) & cities['Population'].apply(lambda X: X > 1000.00))

cities

cities.index

cities.City_name.index

cities = cities.reindex(np.random.permutation(cities.index))
cities


cities = cities.reindex([6,5,4,1,2,0,3])

cities
