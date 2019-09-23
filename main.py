import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, transforms, models


from utils import seed
from utils.executable import Executable
from utils.arguments import Arguments


def main(args: Arguments.parse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = datasets.ImageFolder(args.dataset, transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.2919, 0.2633, 0.2623], [0.1993, 0.2028, 0.1999])
    ]))
    loader = data.DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=4)

    model = models.resnet101(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))

    executor = Executable(args.command)

    executor.init(model, device, path=args.model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model, acc = executor(
        model, loader, criterion, optimizer, scheduler, device=device, num_epochs=args.epoch)

    if executor.command == 'train':
        torch.save(model.state_dict(), args.output)
    elif executor.command == 'eval':
        print(acc)


if __name__ == '__main__':
    arguments = Arguments()
    seed(arguments.seed)
    main(arguments)
