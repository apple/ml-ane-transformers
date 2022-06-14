import torch
import datetime
import coremltools as ct

bar = "="

print("{}".format(bar*13))
times = datetime.datetime.now()
version = torch.__version__
coremltools_version = ct.__version__
device = torch.device("gpu") if torch.cuda.is_available() else "cpu"
print("INFO :{}".format(times))
print("CoreMLTools Version: {}".format(coremltools_version))
print("Pytorch Version:{}".format(version))
print("Device :{}".format(device))
print("{}".format(bar*13))
