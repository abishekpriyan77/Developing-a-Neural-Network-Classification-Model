# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Write your own stepsLoad and Preprocess the Dataset:

Import the required libraries, load the dataset, handle missing values, encode categorical variables, and normalize the features.
### STEP 2: 

Split the Dataset:

Divide the dataset into training and testing sets.

### STEP 3: 

Build the Neural Network:

Create a Sequential model with an input layer, hidden layers using ReLU activation, and an output layer with Softmax activation for the four customer segments.

### STEP 4: 

Train the Model:

Compile the model using the Adam optimizer and categorical cross-entropy loss function, then train it using the training data.

### STEP 5: 

Evaluate and Predict:

Evaluate the model on the test dataset, measure its accuracy, and use it to predict the customer segment (A, B, C, or D) for new customers.

## PROGRAM


```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("customers.csv")
df

df.drop(columns = ['ID'], inplace = True)
df

df.fillna({'Work_Experience' : 0, 'Family_Size': df['Family_Size'].median()}, inplace = True)

df.columns

cat_cols = ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score', 'Var_1','Segmentation']
for  col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])
df

x = df.drop(columns = ['Segmentation'])
y = df['Segmentation'].values
xt,xst,yt,yts = train_test_split(x,y,test_size = 0.2, random_state = 42)

scaler1 = StandardScaler()
scaler2 = StandardScaler()
xt = scaler1.fit_transform(xt)
xst = scaler2.fit_transform(xst)

xt = torch.FloatTensor(xt)
xst = torch.FloatTensor(xst)
yt = torch.FloatTensor(yt)
yst = torch.FloatTensor(yts)

train = TensorDataset(xt, yt)
test = TensorDataset(xst,yst)
trl = DataLoader(train,batch_size=16,shuffle = True)
tsl = DataLoader(test,batch_size = 16)

class classification_model(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, 32)
        self.l2 = nn.Linear(32, 16)
        self.l3 = nn.Linear(16, 8)
        self.l4 = nn.Linear(8, 4)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        return x

         
model = classification_model(xt.shape[1])
criterion = nn.CrossEntropyLoss()
op = optim.Adam(model.parameters(), lr = 0.001)

epochs = 100
for i in range(epochs):
    for a,b in trl:
        pred = model(a)
        loss = criterion(pred,b.long())
        loss.backward()
        op.step()
        if i % 10 == 0:
            print(f"Loss {i + 1} / e Loss: {loss.item()}")

act = []
pre = []
with torch.no_grad():
    for a,b in tsl:
        pred = model(a)
        _,output = torch.max(pred,1)
        pre.extend(output.numpy())
        act.extend(b.numpy())

accuracy = accuracy_score(act,pre)
conf_matrix = confusion_matrix(act,pre)
class_report = classification_report(act,pre,target_names = ['A','B','C','D'])
print("Accuracy = ",accuracy)
print("Confusion Matrix = \n",conf_matrix)
print("Classification Report = \n",class_report)

sns.heatmap(conf_matrix,annot = True,fmt = 'd', cmap = 'Blues', xticklabels = ['A','B','C','D'], yticklabels = ['A','B','C','D'])
plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")
plt.title("Confusion Matrix")
```

### Dataset Information
<img width="1465" height="452" alt="image" src="https://github.com/user-attachments/assets/aee1299e-9f36-4c96-bcff-482d3329df29" />


### OUTPUT
## Accuracy Score
<img width="277" height="33" alt="image" src="https://github.com/user-attachments/assets/e9fd077c-92eb-4743-919c-866626817d74" />


## Confusion Matrix
<img width="193" height="106" alt="image" src="https://github.com/user-attachments/assets/d9efa9f5-5e91-45ee-83a1-3fac68820259" />


## Classification Report
<img width="521" height="233" alt="image" src="https://github.com/user-attachments/assets/34b1085e-f573-4eb2-a8d4-920c326170e0" />

## Heatmap Of Confusion Matrix
<img width="646" height="542" alt="image" src="https://github.com/user-attachments/assets/59b12f72-5160-4eff-ad9d-1a151ccef0e4" />


## RESULT
The program has been executed successfully.
