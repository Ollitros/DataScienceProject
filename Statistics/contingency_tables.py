import pandas as pd
from scipy.stats import chi2_contingency


matrix = pd.DataFrame(data={'man': [20, 11, 7], 'woman': [15, 12, 9]}, index=['Stop', 'Suspend', 'Doesnt stop'])
result = chi2_contingency(observed=matrix)
print(result)