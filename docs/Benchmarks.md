# Benchmarks <a class="anchor" id="benchmarks"></a>

## Diabetic Retinopathy Detection <a class="anchor" id="diabetic-retinopathy-detection"></a>

The first task is based on Leibig's work
_Leveraging Uncertainty Information from Deep Neural Networks for Disease Detection_
published at Nature in 2017 [[4]](../docs/Citations.md#Leibig-2017). Leibig et. al address the task of
reducing burden on the health-care system in diagnosing diabetic retinopathy by using a
dropout-based uncertainty measure to direct 80% of cases to model-based analysis and only
20% of the most uncertain classifications for further review by a physician. The data for
this work consists of fundus imagery along with a binary label indicating the presence of
diabetic retinopathy, a complication of diabetes that if left undiagnosed can eventually
lead to blindness.

### Download <a class="anchor" id="diabetic-retinopathy-detection-download"></a>

The Diabetic Retinopathy Detection dataset is hosted by `Kaggle`, hence you will need a
Kaggle account to fetch it. The `Kaggle` Credentials can be found at
`https://www.kaggle.com/<username>/account -> "Create New API Key"`.

After creating an API key you will need to accept the dataset license. Go to [the dateset page on Kaggle](https://www.kaggle.com/c/diabetic-retinopathy-detection/rules) and look for the button `I Understand and Accept` (make sure when reloading the page that the button does not pop up again).

Provided that `bdlb` is installed, you can automatically setup the environment by running:

```shell
python3 data/setup.py \
  --benchmark=diabetes \
  --kaggle-user=XXXXX \
  --kaggle-key=XXXXXXXXXX
```

which will automatically fetch, unzip and format the data at `./data/diabetes/` (the entire process should take about 4-5 hours to set you up, and only needs to be done once). 

Once the benchmark is set up, you can browse through the example [baselines](../examples/diabetes/) and example [notebooks](../examples/notebooks/).