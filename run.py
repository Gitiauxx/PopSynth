from synthetize import synthetize
import os

# configuration file
yamlfile = '..\\Data\Inputs\config_synthetizer.yaml'

# ouput folder
outdir = 'C:\\Users\\xgitiaux\\Documents\\Research and Analysis\\Synthetizer\\Data\\Outputs'

# synthetizer method
sy = synthetize.from_config(str_or_buffer=yamlfile)

# create data and estimate joint distribution for each block group in Colorado
beta, loss = sy.estimate_distribution()

# report loss
loss = loss[loss > 0]
n = loss.shape[0]
print("after %d iterations the average loss is %f" % (n - 1, loss[n - 1]))

# create a synthetic population
synthHH = sy.draw(beta)
synthHH.to_csv(os.path.join(outdir, 'households_table.csv'))

# create validation metrics
validation_data = sy.validation(synthHH)
validation_data.to_csv(os.path.join(outdir, 'validation_table.csv'))
