# MOMENT forecasting

- Original MOMENT repo [here](https://github.com/moment-timeseries-foundation-model/moment).
- Research repo (with experimentation code) [here](https://github.com/moment-timeseries-foundation-model/moment-research).
- MOMENT paper [here](https://arxiv.org/abs/2402.03885).

The structure of the code references the MOMENT [forecasting tutorial](https://github.com/moment-timeseries-foundation-model/moment/blob/main/tutorials/forecasting.ipynb), and the training/eval code from [Time-Series-Library](https://github.com/thuml/Time-Series-Library/blob/main/tutorial/TimesNet_tutorial.ipynb).

To run:
- Install dependencies in `requirements.txt`
- Change required variables in the `scripts\CS702.sh` script, and run:
```
sh scripts\CS702.sh
```

Currently running linear-probing (as defined in the paper) with MOMENT-large on the dataset for 1 epoch based on MSE loss (last 4 datapoints prediction). In the test dataset, we compute the similarity between the forecasted features (correct order) and the original features (wrong order) for points 11-13, and then permutate accordingly to give us the output. This is then concatenated with the final datapoint prediction (point 14), to form a single row for 1283 datapoints.

### Note
The code currently requires a small change in the MOMENT library to function properly. In the `forecast` method of the `MOMENT` class, add a `.forecast` to the return variable, like so:
``` python
  def forecast(
      self, x_enc: torch.Tensor, input_mask: torch.Tensor = None, **kwargs
  ) -> TimeseriesOutputs:
      batch_size, n_channels, seq_len = x_enc.shape

      # ... MOMENT code ...

      dec_out = self.head(enc_out)  # [batch_size x n_channels x forecast_horizon]
      dec_out = self.normalizer(x=dec_out, mode="denorm")

      return TimeseriesOutputs(input_mask=input_mask, forecast=dec_out).forecast
```
This is so that the existing training/eval code is more in line with that from the Time-Series-Library. All MOMENT methods return a `TimeSeriesOutputs` object, so this simplifies the output from the model _(doesn't require appending `.forecast` on every other output variable)_.
