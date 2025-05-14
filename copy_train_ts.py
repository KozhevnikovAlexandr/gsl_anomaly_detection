import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm.auto import tqdm, trange
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from dataset_swat import ForecastSimpleDataloader
from code_test import test
from copy_transformer import TimeSeriesTransformerGSL

def parse_args():
    parser = argparse.ArgumentParser(description='Train transformer+GSL model for SWaT forecasting')
    parser.add_argument('--normal_csv', type=str, default='SWaT_Dataset_Normal_v0.csv')
    parser.add_argument('--attack_csv', type=str, default='SWaT_Dataset_Attack_v0.csv')
    parser.add_argument('--n_epochs', type=int, default=150)
    parser.add_argument('--window_size', type=int, default=100)
    parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dim_feedforward', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gsl_k', type=int, default=7)
    parser.add_argument('--n_gnn', type=int, default=1)
    parser.add_argument('--n_hidden', type=int, default=256)
    parser.add_argument('--name', type=str, default='swat_transformer_gsl')
    return parser.parse_args()

def train():    
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # 1) Читаем нормальный и атакующий датасеты
    df_norm = pd.read_csv(args.normal_csv)
    df_att  = pd.read_csv(args.attack_csv)

    # 2) Добавляем колонку меток: 0 для нормальных, 1 для атак
    df_norm['Normal/Attack'] = 0

    # Объединяем для теста, а для трейна берем только нормальные
    df_train = df_norm.copy().reset_index(drop=True)
    df_test  = pd.concat([df_norm, df_att], ignore_index=True)

    # 3) Масштабирование признаков — учим только на train
    feature_cols = [c for c in df_train.columns if c not in ['Timestamp', 'Normal/Attack']]
    
    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols])
    df_train[feature_cols] = df_train[feature_cols].astype('float32')
    df_test[feature_cols] = df_test[feature_cols].astype('float32')

    df_train[feature_cols] = scaler.transform(df_train[feature_cols])
    df_test[feature_cols]  = scaler.transform(df_test[feature_cols])

    ts_dim = len(feature_cols)

    # 4) Создаём даталоадеры
    train_dl = ForecastSimpleDataloader(
        dataframe=df_train,
        window_size=args.window_size,
        step_size=args.step_size,
        use_minibatches=True,
        batch_size=args.batch_size,
        shuffle=True,
        data_framework='torch',
        device=device,
        train=True,              # фильтрация по меткам 0
        timestamp_col='Timestamp',
        label_col='Normal/Attack',
        disable_index=False,
        random_state=42
    )

    # test_dl = ForecastSimpleDataloader(
    #     dataframe=df_test,
    #     window_size=args.window_size,
    #     step_size=args.step_size,
    #     use_minibatches=True,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     data_framework='torch',
    #     device=device,
    #     train=False,             # используем все окна, в том числе с атаками
    #     timestamp_col='Timestamp',
    #     label_col='Normal/Attack',
    #     disable_index=False
    # )

    # 5) Модель и оптимизатор
    model = TimeSeriesTransformerGSL(
        ts_dim=ts_dim,
        window_size=args.window_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        gsl_k=args.gsl_k,
        n_gnn=args.n_gnn,
        n_hidden=args.n_hidden,
        device=device
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)

    # 6) Тренировка
    model.train()
    for epoch in trange(args.n_epochs, desc="Epochs"):
        epoch_losses, epoch_smape = [], []
        for ts_batch, _, target_batch, _ in tqdm(train_dl, desc="Batches", leave=False):
            optimizer.zero_grad()
            pred = model(ts_batch)

            loss = F.l1_loss(pred, target_batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

            # считаем SMAPE в оригинальном масштабе
            with torch.no_grad():
                tgt_np  = scaler.inverse_transform(target_batch.cpu().numpy())
                pred_np = scaler.inverse_transform(pred.cpu().numpy())
                num = np.abs(tgt_np - pred_np)
                den = np.abs(tgt_np) + np.abs(pred_np) + 1e-8
                epoch_smape.append((num/den).mean()*100)

        print(f"Epoch {epoch+1}/{args.n_epochs} — MAE: {np.mean(epoch_losses):.4f}, SMAPE: {np.mean(epoch_smape):.2f}%")

    # # 7) Тестирование
    # model.eval()
    # test_metrics = test(
    #     model=model,
    #     test_dl=test_dl,
    #     scaler=scaler,
    #     dataset_obj=None,             # можно передать ваш объект, если нужен
    #     use_threshold_from_val=False  # или True, если есть валидационные метки
    # )
    # print("Test metrics:", test_metrics)

    # 8) Сохраняем
    os.makedirs('saved_models', exist_ok=True)
    torch.save(model.state_dict(), f"saved_models/{args.name}3.pt")


if __name__ == '__main__':
    train()
