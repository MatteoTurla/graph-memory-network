from model.GMNclassification import GMNclassification
from torch_geometric.data import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset
import torch

from ogb.graphproppred import Evaluator


def train(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = total_batch = 0.0
    total_example = total_correct = 0.0

    for data in loader:
        data = data.to(device)
        x = data.x.float()
        edge_index = data.edge_index

        # in case y is a n x 1 tensor (ogb graph)
        y = data.y.squeeze()

        out = model(x, edge_index, data.batch)

        optimizer.zero_grad()

        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batch += 1
        total_example += out.shape[0]
        total_correct += out.argmax(dim=-1).eq(y).sum().item()

    return total_loss/total_batch, total_correct/total_example


@torch.no_grad()
def test(model, loader, device):
    model.eval()

    total_example = total_correct = 0.0

    y_true = torch.Tensor([[]])
    y_pred = torch.Tensor([[]])
    for data in loader:
        x = data.x.float().to(device)
        edge_index = data.edge_index.to(device)
        batch = data.batch.to(device)

        # in case y is a n x 1 tensor (ogb graph)
        y = data.y.squeeze().cpu()

        out = model(x, edge_index, batch).cpu()

        total_example += out.shape[0]
        total_correct += out.argmax(dim=-1).eq(y).sum().item()

        y_true = torch.vstack(y_true, y)
        y_pred = torch.vstack(y_pred, out.argmax(dim=-1))

    return total_correct/total_example, {"y_true": y_true, "y_pred": y_pred}


if __name__ == "__main__":

    # Download and process data at './dataset/ogbg_molhiv/'
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root='dataset/')

    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(
        dataset[split_idx["train"]], batch_size=128, shuffle=True)
    valid_loader = DataLoader(
        dataset[split_idx["valid"]], batch_size=128, shuffle=False)
    test_loader = DataLoader(
        dataset[split_idx["test"]], batch_size=128, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    model = GMNclassification(
        dataset.num_node_features, dataset.num_classes, 1, 5).to(device)
    # print(model.train())

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.0)
    criterion = torch.nn.CrossEntropyLoss()

    for e in range(1):
        train_loss, train_acc = train(
            model, train_loader, criterion, optimizer, device)
        if e % 10 == 0:
            loss_acc, dict_evaluator = test(model, valid_loader, device)
            print(train_loss, train_acc, loss_acc)

    evaluator = Evaluator(name="ogbg-molhiv")
    result_dict = evaluator.eval(dict_evaluator)
