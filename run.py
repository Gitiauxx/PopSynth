from synthetize import synthetize

# configuration file
yamlfile = '..\\Data\Inputs\config_synthetizer.yaml'

# ouput folder

# synthetizer method
sy = synthetize.from_config(str_or_buffer=yamlfile)

# create data and estimate joint distribution for each block group in Colorado
beta, loss = sy.estimate_distribution()

# report loss
loss = loss[loss > 0]
n = loss.shape[0]
print("after %d iterations the average loss is %f" % (n - 1, loss[n - 1]))

# create a synthetic population
synthHH = sy.draw()
synthHH.to_csv()
