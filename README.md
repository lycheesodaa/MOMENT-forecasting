# MOMENT forecasting

- Original MOMENT repo [here](https://github.com/moment-timeseries-foundation-model/moment).
- Research repo (with experimentation code) [here](https://github.com/moment-timeseries-foundation-model/moment-research).
- MOMENT paper [here](https://arxiv.org/abs/2402.03885).

The structure of the code references the MOMENT [forecasting tutorial](https://github.com/moment-timeseries-foundation-model/moment/blob/main/tutorials/forecasting.ipynb), and the training/eval code from [Time-Series-Library](https://github.com/thuml/Time-Series-Library/blob/main/tutorial/TimesNet_tutorial.ipynb).

Currently tuned for the Stock and Demand datasets. Add more dataloaders and `run_xx.py` scripts as necessary (though most datasets should follow the current `run.py` workflow, which is to:
1. pre-evaluate the out-of-the-box MOMENT model,
2. train the MOMENT model either through linear probing (1 pass) or earlystopping (with validation set), and
3. evaluate the trained model.
