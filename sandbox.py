import pandas as pd

df = pd.read_csv('job_postings.csv')
descrs = df['description'].to_list()

print(descrs[0])

