# src/train.py
import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from dataset import WeldingDataset
from model import PhysFusionNet
import pandas as pd

def train():
    # ⏱️ 记录训练开始时间
    start_time = time.time()

    # 路径设定
    csv_path = os.path.join(r'F:\OneDrive\WorkSpace\FSW_Thermal_Correction\data\main.csv')
    label_dir = os.path.join(r'F:\OneDrive\WorkSpace\FSW_Thermal_Correction\data')
    images_dir = os.path.join(r'F:\OneDrive\WorkSpace\FSW_Thermal_Correction\images')
    
    dataset = WeldingDataset(csv_path, label_dir, images_dir)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysFusionNet(param_dim=3).to(device)

    # ▼选择优化器（默认使用Adam）
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
    # optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # optimizer = optim.Adagrad(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # optimizer = optim.RMSprop(model.parameters(), lr=1e-4, alpha=0.99, weight_decay=1e-5)
    # optimizer = optim.Adamax(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # optimizer = optim.ASGD(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # optimizer = optim.LBFGS(model.parameters(), lr=1e-4, max_iter=20, history_size=10, line_search_fn='strong_wolfe')
    # optimizer = optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, weight_decay=1e-5)

    criterion = torch.nn.MSELoss()
    epoch_losses = []
    num_epochs = 100

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch in dataloader:
            # 工艺参数向量：[rpm, speed, h]
            params = torch.stack([
                torch.tensor(batch['rpm']),
                torch.tensor(batch['speed']),
                torch.tensor(batch['h'])
            ], dim=1).float().to(device)

            roi = torch.tensor(batch['roi']).unsqueeze(1).float().to(device)  # [B, 1, 64, 64]
            tc = torch.tensor(batch['tc'][:, 0]).unsqueeze(1).float().to(device)
            T_env_seq = torch.zeros(roi.size(0), 5, 1).float().to(device)

            optimizer.zero_grad()
            T_corr, physics_loss = model(roi, params, T_env_seq)
            loss_reg = criterion(T_corr, tc)
            loss = loss_reg + 0.1 * physics_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "phys_fusion_model.pth")
    print("✅ Training completed!")

    # 统计并打印训练总耗时
    end_time = time.time()
    elapsed = end_time - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"🕒 Total training time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

    # 可选：保存损失记录
    loss_df = pd.DataFrame({
        'Epoch': range(1, num_epochs + 1),
        'Loss': epoch_losses
    })
    loss_df.to_csv('training_losses_adam.csv', index=False)

if __name__ == "__main__":
    train()
