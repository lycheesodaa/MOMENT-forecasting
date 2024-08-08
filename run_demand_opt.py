import argparse
import os
import random
import time
from tqdm import tqdm

import numpy as np
import torch
from torch.nn import DataParallel
from torch.optim.lr_scheduler import OneCycleLR

from momentfm import MOMENTPipeline
from data_provider.data_factory import data_provider


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model(args, device):
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={
            'task_name': 'forecasting',
            'forecast_horizon': args.pred_len
        },
    )
    model.init()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)
    return model.to(device)


def val_or_test(model, loader, criterion, mae_metric, device, args, target_index):
    model.eval()
    total_loss = []
    total_mae_loss = []
    total_forecast_loss = []
    total_forecast_mae_loss = []

    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader)):
            batch_x, batch_y, _, _, nems_forecast_x, nems_forecast_y, input_mask = [b.to(device) for b in batch]

            pred = model(batch_x, input_mask)

            pred = pred[:, target_index, :]
            true = batch_y[:, target_index, :]

            loss = criterion(pred, true)
            mae_loss = mae_metric(pred, true)

            total_loss.append(loss.item())
            total_mae_loss.append(mae_loss.item())

            forecasts = nems_forecast_y
            forecast_loss = criterion(forecasts, true)
            forecast_mae_loss = mae_metric(forecasts, true)

            total_forecast_loss.append(forecast_loss.item())
            total_forecast_mae_loss.append(forecast_mae_loss.item())

    return map(np.mean, [total_loss, total_mae_loss, total_forecast_loss, total_forecast_mae_loss])


def train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device, args, target_index):
    model.train()
    total_loss = []

    for batch in tqdm(train_loader, total=len(train_loader)):
        batch_x, batch_y, _, _, _, _, input_mask = [b.to(device) for b in batch]

        optimizer.zero_grad(set_to_none=True)

        if args.use_amp:
            with torch.cuda.amp.autocast():
                output = model(batch_x, input_mask)
                loss = criterion(output[:, target_index, :], batch_y[:, target_index, :])

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(batch_x, input_mask)
            loss = criterion(output[:, target_index, :], batch_y[:, target_index, :])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

        scheduler.step()
        total_loss.append(loss.item())

    return np.mean(total_loss)


def main():
    parser = argparse.ArgumentParser(description='MOMENT')
    # Add all your arguments here
    # ...
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='max gradient norm for clipping')
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model, criterion, and metrics setup
    model = get_model(args, device)
    criterion = torch.nn.MSELoss().to(device)
    mae_metric = torch.nn.L1Loss().to(device)

    # Data loading
    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    # Optimizer and scheduler setup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = OneCycleLR(optimizer, max_lr=args.learning_rate, total_steps=len(train_loader) * args.train_epochs,
                           pct_start=args.pct_start)

    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    target_index = 0  # Assuming this is always 0

    # Training loop
    for epoch in range(args.train_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device, args,
                                 target_index)
        print(f"Epoch {epoch + 1}/{args.train_epochs}: Train loss: {train_loss:.3f}")

        vali_loss, vali_mae_loss, vali_forecast_loss, vali_forecast_mae_loss = val_or_test(model, vali_loader,
                                                                                           criterion, mae_metric,
                                                                                           device, args, target_index)
        test_loss, test_mae_loss, test_forecast_loss, test_forecast_mae_loss = val_or_test(model, test_loader,
                                                                                           criterion, mae_metric,
                                                                                           device, args, target_index)

        print(f"Validation - Loss: {vali_loss:.7f}, MAE: {vali_mae_loss:.7f}")
        print(f"Test - Loss: {test_loss:.7f}, MAE: {test_mae_loss:.7f}")
        print(f"NEMS Forecast - Validation Loss: {vali_forecast_loss:.7f}, MAE: {vali_forecast_mae_loss:.7f}")
        print(f"NEMS Forecast - Test Loss: {test_forecast_loss:.7f}, MAE: {test_forecast_mae_loss:.7f}")

    # Save the final model
    torch.save(model.state_dict(), os.path.join(args.checkpoints, f'{args.model_id}_final_model.pth'))


if __name__ == '__main__':
    main()