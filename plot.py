import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import pathlib

filepath = os.path.join(pathlib.Path(__file__).parent.resolve(), 'results.csv')
df = pd.read_csv(filepath)

# print the first 5 rows
print(df.head())

# plot column 1 as 'loss/policy'
plt.plot(df.iloc[:, 0], label='loss/policy')
plt.xlabel('Timestep')
plt.ylabel('Loss')
plt.title('Loss/Policy Over Time')
plt.legend()
plt.savefig('loss_policy.png', dpi=300, bbox_inches='tight')
plt.show()

# plot column 2 as 'loss/value'
plt.plot(df.iloc[:, 1], label='loss/value')
plt.xlabel('Timestep')
plt.ylabel('Loss')
plt.title('Loss/Value Over Time')
plt.legend()
plt.savefig('loss_value.png', dpi=300, bbox_inches='tight')
plt.show()

# plot column 3 as 'loss/entropy'
plt.plot(df.iloc[:, 2], label='loss/entropy')
plt.xlabel('Timestep')
plt.ylabel('Loss')
plt.title('Loss/Entropy Over Time')
plt.legend()
plt.savefig('loss_entropy.png', dpi=300, bbox_inches='tight')
plt.show()

# plot column 5 as 'policy/clip_fraction'
plt.plot(df.iloc[:, 4], label='policy/clip_fraction')
plt.xlabel('Timestep')
plt.ylabel('Fraction')
plt.title('Policy/Clip Fraction Over Time')
plt.legend()
plt.savefig('policy_clip_fraction.png', dpi=300, bbox_inches='tight')
plt.show()

# plot column 6 as 'policy/ratio'
plt.plot(df.iloc[:, 5], label='policy/ratio')
plt.xlabel('Timestep')
plt.ylabel('Ratio')
plt.title('Policy Ratio Over Time')
plt.legend()
plt.savefig('policy_ratio.png', dpi=300, bbox_inches='tight')
plt.show()
