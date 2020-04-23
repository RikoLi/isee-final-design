from matplotlib import pyplot as plt
import numpy as np

with open('../../logs/000/train_loss.csv', 'r') as f:
    lines = f.readlines()
steps = []
train_loss = []
for line in lines[1:]:
    line = line.replace('\n', '')
    steps.append(float(line.split(',')[1]))
    train_loss.append(float(line.split(',')[2]))
steps = np.array(steps)
train_loss = np.array(train_loss)

with open('../../logs/000/val_loss.csv', 'r') as f:
    lines = f.readlines()
val_steps = []
val_loss = []
for line in lines[1:]:
    line = line.replace('\n', '')
    val_steps.append(float(line.split(',')[1]))
    val_loss.append(float(line.split(',')[2]))
val_steps = np.array(val_steps)
val_loss = np.array(val_loss)

# print(val_loss)

train_loss = np.log10(train_loss)
val_loss = np.log10(val_loss)

plt.plot(steps, train_loss, color='red', label='YOLOv3 train loss', marker='.', linewidth='0.5', markersize='2')
plt.plot(val_steps, val_loss, color='blue', label='YOLOv3 val loss', marker='.', linewidth='0.5', markersize='2')
plt.title('Training Curve')
plt.xlabel('Iterations')
plt.ylabel('Loss (in log10 scale)')
plt.legend()
plt.grid(axis='y', linestyle='--')
plt.savefig('training_curve.png', dpi=600)
plt.show()