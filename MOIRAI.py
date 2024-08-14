import torch
import matplotlib.pyplot as plt
import pandas as pd
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from huggingface_hub import hf_hub_download

from uni2ts.eval_util.plot import plot_single, plot_next_multi
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule


SIZE = "small"  # model size: choose from {'small', 'base', 'large'}
PDT = 72  # prediction length: any positive integer
CTX = 512  # context length: any positive integer
PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
BSZ = 16  # batch size: any positive integer

# data formatting into GluonTS dataset
df = pd.read_csv('~/models/MOMENT/data/demand_data_all_cleaned_numerical.csv', index_col=0, parse_dates=True)

train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

total_len = len(df)
train_len = int(train_ratio * total_len)
val_len = int(val_ratio * total_len)
test_len = int(test_ratio * total_len)

border1s = [0, train_len - CTX, train_len + val_len - CTX]
border2s = [train_len, train_len + val_len, train_len + val_len + test_len]

train_df = df[border1s[0]:border2s[0]]
val_df = df[border1s[1]:border2s[1]]
test_df = df[border1s[2]:border2s[2]]

train_ds = PandasDataset(dict(train_df))
val_ds = PandasDataset(dict(val_df))
test_ds = PandasDataset(dict(test_df))

# Group time series into multivariate dataset
grouper = MultivariateGrouper(train_len)
multivar_train_ds = grouper(train_ds)
grouper = MultivariateGrouper(val_len)
multivar_val_ds = grouper(val_ds)
grouper = MultivariateGrouper(test_len)
multivar_test_ds = grouper(test_ds)

train, test_template = split(
    multivar_test_ds, offset=-test_len-CTX
)  # assign last TEST time steps as test set

# evaluation
# Construct rolling window evaluation
test_data = test_template.generate_instances(
    prediction_length=PDT,  # number of time steps for each prediction
    windows=test_len - PDT + 1,  # number of windows in rolling window evaluation (TEST//PDT for non-overlap)
    distance=1,  # number of time steps between each window (distance=PDT for non-overlapping windows)
)

# Prepare pre-trained model by downloading model weights from huggingface hub
model = MoiraiForecast(
    module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{SIZE}"),
    prediction_length=PDT,
    context_length=CTX,
    patch_size=PSZ,
    num_samples=100,
    target_dim=len(test_ds),
    # feat_dynamic_real_dim=test_ds.num_feat_dynamic_real,
    # past_feat_dynamic_real_dim=test_ds.num_past_feat_dynamic_real,
    feat_dynamic_real_dim=0,
    past_feat_dynamic_real_dim=0,
)

predictor = model.create_predictor(batch_size=BSZ, device='cuda:0')
forecasts = predictor.predict(test_data.input)

input_it = iter(test_data.input)
label_it = iter(test_data.label)
forecast_it = iter(forecasts)

inp = next(input_it)
label = next(label_it)
forecast = next(forecast_it)

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(25, 10))
for i, ax in enumerate(axes.flatten()):
    plot_single(
        inp,
        label,
        forecast,
        context_length=512,
        intervals=(0.5, 0.9),
        dim=i,
        ax=ax,
        name="pred",
        show_label=True,
    )
plt.show()