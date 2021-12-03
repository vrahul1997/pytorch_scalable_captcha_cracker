from tqdm import tqdm
import torch
import config


def train(model, dataloader, optimizer):
    model.train()
    fn_loss = 0
    tk = tqdm(dataloader, total=len(dataloader))
    for data in tk:
        for k, v in data.items():
            data[k] = v.to(config.DEVICE)
        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        optimizer.step()
        fn_loss += loss.item()
    return fn_loss / len(dataloader)


def eval(model, dataloader):
    model.eval()
    fn_loss = 0
    fn_preds = []
    tk = tqdm(dataloader, total=len(dataloader))
    with torch.no_grad():
        for data in tk:
            for k, v in data.items():
                data[k] = v.to(config.DEVICE)
            batch_preds, loss = model(**data)
            fn_loss += loss.item()
            fn_preds.append(batch_preds)
    return fn_preds, fn_loss / len(dataloader)
