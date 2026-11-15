from config import Config
from preprocessing import build_dataloader
from bert_model import BertClassifierModel
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

config = Config()

def eval_model(model):
    # 加载验证集
    dev_dataloader = build_dataloader(datapath=config.dev_datapath,shuffle=False)

    all_pred_result = []    # 预测结果列表
    all_true_result = []    # 真实结果列表

    model.eval()
    with torch.no_grad():
        for i,batch in enumerate(tqdm(dev_dataloader), start=1):
           
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(config.device)
            attention_mask = attention_mask.to(config.device)
            labels = labels.to(config.device)

            # 前向传播
            pred_output = model(input_ids,attention_mask)
            pred_index = torch.argmax(pred_output,dim=-1)
            
            all_pred_result.extend(pred_index.cpu().tolist())
            all_true_result.extend(labels.cpu().tolist())

    
    f1score = f1_score(all_true_result,all_pred_result,average="macro")
    # 准确率
    accuracy = accuracy_score(all_true_result,all_pred_result)
    precision = precision_score(all_true_result,all_pred_result,average="macro")
    recall = recall_score(all_true_result,all_pred_result,average="macro")

    return f1score, accuracy, precision, recall

def train_and_eval():
    # 加载数据
    train_dataloader = build_dataloader(datapath=config.train_datapath,shuffle=True)

    # 创建模型
    model = BertClassifierModel().to(device=config.device)

    # 损失函数对象
    loss = nn.CrossEntropyLoss()

    # 优化器对象
    optimizer = torch.optim.AdamW(params=model.parameters(),lr=5e-5)

    epochs = 3

    # 训练
    model.train()   # 切换为训练模式
    best_f1score = 0.0  # f1score历史最高分

    for epoch in range(epochs):
        total_loss = 0.0    # 总损失值

        for i,batch in enumerate(tqdm(train_dataloader),start=1):

            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(config.device)
            attention_mask = attention_mask.to(config.device)
            labels = labels.to(config.device)

            # 前向传播
            pred_output = model(input_ids,attention_mask)

            # 算损失值
            loss_value = loss(pred_output, labels)
            total_loss += loss_value.item()

            optimizer.zero_grad()
            loss_value.backward() 
            optimizer.step()

            # 每隔100个批次对已训练的模型进行验证
            if i%100==0 or i==len(train_dataloader):
                # 调用评估函数
                f1score, accuracy, precision, recall = eval_model(model)
                print(f"第{i}批次，f1score={f1score}，accuracy={accuracy}，precision={precision}，recall={recall}")

             
                if f1score>best_f1score:
                    os.makedirs(os.path.dirname(config.save_model), exist_ok=True)
                    torch.save(model.state_dict(),config.save_model)
                    best_f1score = f1score  # 更新历史最高分

                # 为训练模式
                model.train()

if __name__ == '__main__':
    train_and_eval()
