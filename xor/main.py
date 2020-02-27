import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

class XORNets(nn.Module):
	def __init__(self, args):
		super(XORNets, self).__init__()

		self.args = args
		self.device = torch.device("cuda" if args.cuda else "cpu")

		self.layer = nn.Sequential(
		nn.Linear(2, 2),
		nn.ReLU(),
		nn.Linear(2, 1),
		nn.Sigmoid()
		)

		self.weights_init()

	def forward(self, x):
		return self.layer(x)

	def train_(self, optimizer):
		data = torch.from_numpy(x).float()
		label = torch.from_numpy(y).float()

		for epoch in range(1, self.args.epochs + 1):
			data = data.to(self.device)
			label = label.to(self.device)

			output = self(data)

			optimizer.zero_grad()
			loss = F.mse_loss(output, label)
			loss.backward()
			optimizer.step()
		
			if epoch % 100 == 0:
				print('\n')
				print('Epoch: {}'.format(epoch))
				print('Loss: {:.4f}'.format(loss))

	def predict_(self):
		with torch.no_grad():
			data = torch.from_numpy(x).float().to(self.device)
			label = y
			output = self(data)

			for i in range(len(data)):
				print("Input:", data.data.cpu().numpy().astype(np.int32)[i])
				print("Label:", label[i])
				print("Predicted output:", output.data.cpu().numpy()[i])

	def weights_init(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 1)
				m.bias.data.zero_()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--cuda", action="store_true", help="Using Cuda")
	parser.add_argument("--lr", type=float, default=0.02, help="Learning rate for SGD optimizer")
	parser.add_argument("--epochs", type=int, default=5000, help="Number of training epochs")
	args = parser.parse_args()

	device = torch.device("cuda" if args.cuda else "cpu")

	model = XORNets(args).to(device)
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

	model.train()
	model.train_(optimizer)
	print("\n")
	model.eval()
	model.predict_()
	print("\n")
