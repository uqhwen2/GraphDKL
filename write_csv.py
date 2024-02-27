import csv
import numpy as np
# Create the CSV file
with open('BlogCatalog_0.5_test.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['']+[round(x, 2) for x in list(np.arange(0, 1, 0.05))])
    writer.writerow(['WithOSN']+[0 for _ in list(np.arange(0, 1, 0.05))])
    writer.writerow(['WithSN']+[0 for _ in list(np.arange(0, 1, 0.05))])
