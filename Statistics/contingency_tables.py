import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact


matrix = pd.DataFrame(data={'man': [20, 11, 7], 'woman': [15, 12, 9]}, index=['Stop', 'Suspend', 'Doesnt stop'])
result = chi2_contingency(observed=matrix)
print(result)

matrix = pd.DataFrame(data={'man': [20, 11], 'woman': [15, 12]}, index=['Stop', 'Suspend'])
result = fisher_exact(table=matrix)
print(result)