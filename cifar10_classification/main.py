import argparse
import random
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def train(model, optimizer, data_loader, epoch, device):
	model.train()
	losses = 0.0
	correct = 0
	total = 0
	for iteration, batch in enumerate(data_loader):
		img = batch[0].to(device)
		target = batch[1].to(device)

		optimizer.zero_grad()
		output = model(img)
		loss = F.cross_entropy(output, target)
		loss.backward()
		optimizer.step()
		losses += loss.item()

		pred = output.data.max(dim=1)[1]
		correct += pred.eq(target.data).sum()
		total += target.size(0)

		if iteration % 100 == 0:
			print('Epoch [{}/{}]\tIteration [{}/{}]\tLoss: {:.4f}'.format(
			epoch, args.epochs, iteration + 1, len(data_loader), losses / (iteration + 1)))
	accuracy = (correct * 100.) / total
	print('Epoch {} training set accuracy: {:.2f}%'.format(epoch, accuracy.item()))

def test(model, data_loader, device):
	model.eval()
	correct = 0
	total = 0
	with torch.no_grad():
		for batch in data_loader:
			img = batch[0].to(device)
			target = batch[1].to(device)

			output = model(img)

			pred = output.data.max(dim=1)[1]
			correct += pred.eq(target.data).sum()
			total += target.size(0)

		accuracy = (correct * 100.) / total
		print('Test set accuracy: {:.2f}%'.format(accuracy.item()))
		print('\n')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', type=int, default=64, help='Set batch size')
	parser.add_argument('--test', action='store_true', help='Only test using trainedmodel')
	parser.add_argument('--lr', type=float, default=0.1, help='Set learning rate for Adam optimizer')
	parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
	parser.add_argument('--cuda', action='store_true', help='Using GPU acceleration')
	parser.add_argument('--threds', type=int, default=4, help='Number of threds for data loading')
	parser.add_argument('--seed', type=int)
	parser.add_argument('--log_dir', default='logs', help='Folder path to save the model')
	parser.add_argument('--checkpoint', default=None, help='File path to load the saved weights')
	args = parser.parse_args()

	if args.seed is None:
		args.seed = random.randint(0, 2 ** 31 - 1)

	if args.test is False and not os.path.exists(args.log_dir):
		os.makedirs(args.log_dir)

	device = torch.device('cuda' if args.cuda else 'cpu')

	torch.manual_seed(args.seed)

	transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	train_loader = DataLoader(
	datasets.CIFAR10(root='data', train=True, transform=transforms, download=True),
	batch_size=args.batch_size,
	shuffle=True,
	num_workers=args.threds
	)
	test_loader = DataLoader(
	datasets.CIFAR10(root='data', train=False, transform=transforms),
	batch_size=args.batch_size,
	shuffle=False,
	num_workers=args.threds
	)

	model = models.resnet18()
	model.fc = nn.Linear(512, 10)
	model = model.to(device)
	if args.checkpoint is not None:
		state_dict = torch.load(args.checkpoint)
		model.load_state_dict(state_dict, strict=False)
	
	if args.test is False:
		optimizer = optim.Adam(model.parameters(), lr=args.lr)

		for epoch in range(1, args.epochs + 1):
			train(model, optimizer, train_loader, epoch, device)
			test(model, test_loader, device)
	
		torch.save(model.state_dict(), os.path.join(args.log_dir, 'epoch_{:02d}.pth'.format(args.epochs)))
	
	else:
		test(model, test_loader, device)
