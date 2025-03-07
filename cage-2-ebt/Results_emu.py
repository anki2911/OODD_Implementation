import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




emu_data = pd.read_csv('./emu_data/bline_emu_ood.csv')

emu_data_col_1 = emu_data[['Iteration']]
emu_data_col_2 = emu_data[['ID']]
emu_data_col_3 = emu_data[['ONT']]
emu_data_col_4 = emu_data[['TNT']]
emu_data_col_5 = emu_data[['ANF']]
emu_data_col_6 = emu_data[['SNF']]

# Create a scatter plot
ax = plt.axes()
plt.scatter(emu_data_col_1, emu_data_col_2, c="green")
plt.scatter(emu_data_col_1, emu_data_col_3, c="green")
plt.scatter(emu_data_col_1, emu_data_col_4, c="red")
plt.scatter(emu_data_col_1, emu_data_col_5, c="red")
plt.scatter(emu_data_col_1, emu_data_col_6, c="red")
plt.rcParams.update({"text.usetex": True})
plt.grid(True)
ax.set_xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
ax.set_xlabel(['0', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50'])
#ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
ax.set_yticks([0, 1, 2, 3])

#ax.set_yticklabels(['', 'In Distribution', 'Mispredicted $S_t$', 'No Transition\n' + 'from $S_{t-1}$ to $S_t$', '$A_{t-1}$ not\n' + 'in Training data', '$S_t$ not\n' + 'in Training data', ''])
#ax.set_yticklabels(['', 'In Distribution\n (Correct Prediction)', 'In Distribution\n ($S_t^{emu}$ different\n from $S_t^{sim}$', 'Out of Distribution\n (Transition from\n $S_{t-1}$ to $S_t$ not\n in Training data)', 'Out of Distribution\n ($A_{t-1}$ not in\n Training data)', ''])
ax.set_yticklabels(['', 'In Distribution\n (Correct Prediction)', 'Out of Distribution\n ($A_{t-1}$ not in\n Training data)', 'Out of Distribution\n ($S_{t-1}$ not in\n Training data)'])


plt.xlabel('Number of Steps',fontsize=14)
plt.ylabel('System Behavior',fontsize=14)
plt.title('Emulation Behavior with Agent trained in Simulation',fontsize=14)


# Show the plot
plt.show()