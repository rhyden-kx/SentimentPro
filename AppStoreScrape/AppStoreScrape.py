
import pandas as pd
import numpy as np
import json

from app_store_scraper import AppStore
GXS = AppStore(country='sg', app_name='GXS Bank', app_id = '1632183616')

GXS.review(how_many=1000)

GXS.reviews

GXSdf = pd.DataFrame(np.array(GXS.reviews),columns=['review'])
GXSdf2 = GXSdf.join(pd.DataFrame(GXSdf.pop('review').tolist()))
GXSdf2.head()

print(GXSdf2)

GXSdf2.to_csv('GXS-app-reviews.csv')

GXSdf2.to_csv('/Users/neleht./Library/CloudStorage/OneDrive-NationalUniversityofSingapore/NUS ALL/Y3S2/DSA3101/Backend/AppStoreData.csv')