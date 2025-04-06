import pandas as pd

midCycleRedistrictingSet = {
    ('AL', 2024),
    ('FL', 2016),
    ('GA', 2024),
    ('LA', 2024),
    ('NY', 2022),
    ('NY', 2024),
    ('NC', 2016),
    ('NC', 2020),
    ('NC', 2022),
    ('NC', 2024),
    ('PA', 2018),
    ('VA', 2018)
}

educationData = pd.read_csv('C:/Users/finco/Documents/GitHub/DS496Capstone/Processed Data/compressedEducationData.csv')
raceData = pd.read_csv('C:/Users/finco/Documents/GitHub/DS496Capstone/Processed Data/raceData.csv')
electionData = pd.read_csv('C:/Users/finco/Documents/GitHub/DS496Capstone/Processed Data/electionLabels.csv')

data = pd.merge(educationData, raceData, how='inner', on=['state_po', 'district', 'year'])
data = pd.merge(data, electionData, how='inner', on=['state_po', 'district', 'year'])

filteredData = data[~data[['state_po', 'year']].apply(tuple, axis=1).isin(midCycleRedistrictingSet)]

filteredData.to_csv('C:/Users/finco/Documents/GitHub/DS496Capstone/Processed Data/finalData.csv', index = False)