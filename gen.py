import csv
import numpy as np
with open("weight-height.csv", encoding="utf-8-sig") as csvfile:
    reader = csv.DictReader(csvfile)
    ww = []
    for row in reader:
        ww.append(row["Weight"])
with open("weight-height.csv", encoding="utf-8-sig") as csvfile:
    reader = csv.DictReader(csvfile)
    hh = []
    for row in reader:
        hh.append(row["Height"])
with open("weight-height.csv", encoding="utf-8-sig") as csvfile:
    reader = csv.DictReader(csvfile)
    gg = []
    for row in reader:
        gg.append(row["Gender"])
for i in range(len(gg)):
    if(gg[i] == "Female"):
        gg[i] = 1
    else:
        gg[i] = 0
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def deriv_sigmoid(x):
  fx = sigmoid(x)
  return fx * (1 - fx)
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()
class OurNeuralNetwork:
  def __init__(self):
    self.w1 = np.random.normal()
    self.w2 = np.random.normal()
    self.w3 = np.random.normal()
    self.w4 = np.random.normal()
    self.w5 = np.random.normal()
    self.w6 = np.random.normal()
    self.b1 = np.random.normal()
    self.b2 = np.random.normal()
    self.b3 = np.random.normal()
  def feedforward(self, x):
    h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
    h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
    o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
    return o1
  def train(self, data, all_y_trues):
    learn_rate = 0.00001
    epochs = 500
    for epoch in range(epochs):
      for x, y_true in zip(data, all_y_trues):
        sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
        h1 = sigmoid(sum_h1)
        sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
        h2 = sigmoid(sum_h2)
        sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
        o1 = sigmoid(sum_o1)
        y_pred = o1
        d_L_d_ypred = -2 * (y_true - y_pred)
        d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
        d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
        d_ypred_d_b3 = deriv_sigmoid(sum_o1)
        d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
        d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)
        d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
        d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
        d_h1_d_b1 = deriv_sigmoid(sum_h1)
        d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
        d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
        d_h2_d_b2 = deriv_sigmoid(sum_h2)
        self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
        self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
        self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1
        self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
        self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
        self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2
        self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
        self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
        self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3
      if epoch % 1 == 0:
        y_preds = np.apply_along_axis(self.feedforward, 1, data)
        loss = mse_loss(all_y_trues, y_preds)
for w in range(len(ww)):
    ww[w] = float(ww[w]) - 140
for h in range(len(hh)):
    hh[h] = float(hh[h]) - 65
data = [None]*len(ww)
for ytr in range(len(ww)):
    data[ytr] = ww[ytr],hh[ytr]
all_y_trues = gg
network = OurNeuralNetwork()
network.train(data, all_y_trues)
def gender(q):
    if(network.feedforward(q)>=0.5):
        a = "Female"
    else:
        a = "Male"
    return a
name = input("Enter Your Name :- ")
take = list(map(float,input("\nEnter the Height and Weight : ").strip().split()))[:2]    
take[0] = take[0] - 140
take[1] = take[1] - 65
print(name," :-",gender(take))
print(network.feedforward(take))
