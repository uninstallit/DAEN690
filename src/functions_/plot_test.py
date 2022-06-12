import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches


#group bar chart
labels = ['G1', 'G2', 'G3', 'G4', 'G5']
men_means = [20, 34, 30, 35, 27]
women_means = [25, 32, 34, 20, 25]


x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, men_means, width, label='Men')
# rects2 = ax.bar(x + width/2, women_means, width, label='Women')

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Scores')
# ax.set_title('Scores by group and gender')
# ax.set_xticks(x, labels)
# ax.legend()

# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)
# fig.tight_layout()
# plt.show()


# stacked bar chart
# labels = ['G1', 'G2', 'G3', 'G4', 'G5']
# men_means = [20, 35, 30, 35, 27]
# women_means = [25, 32, 34, 20, 25]
# width = 0.35       # the width of the bars: can also be len(x) sequence

# fig, ax = plt.subplots()

# ax.bar(labels, men_means, width,  label='Men')
# ax.bar(labels, women_means, width,  bottom=men_means,
#        label='Women')

# ax.set_ylabel('Scores')
# ax.set_title('Scores by group and gender')
# ax.legend()
# plt.show()

##############################################################
fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))

# 1 seaborn stacked graphs
labels = ['G1', 'G2', 'G3', 'G4', 'G5']
men = {}
women ={}
men['scores'] = [20, 34, 30, 35, 27]
women['scores'] = [25, 32, 34, 20, 25]
women['gender'] = labels
men['gender'] = labels
sns.barplot(x=women['gender'], y=women['scores'],  color='lightblue', ax=ax1)
sns.barplot(x=men['gender'], y=men['scores'], color='darkblue', ax=ax1)

ax1.set_ylabel('Scores')
ax1.set_title('Seaborn - Scores by group and gender- row 1-1')
ax1.set_xticks(x, labels)

top_bar = mpatches.Patch(color='lightblue', label='women')
bottom_bar = mpatches.Patch(color='darkblue', label='men')
ax1.legend(handles=[top_bar, bottom_bar])

#2 stacked bar graphs
labels = ['G1', 'G2', 'G3', 'G4', 'G5']
men_means = [20, 34, 30, 35, 27]
women_means = [25, 32, 34, 20, 25]
ax2.bar(labels, men_means, width,  label='Men')
ax2.bar(labels, women_means, width,  bottom=men_means, label='Women')
ax2.set_ylabel('Scores')
ax2.set_title('Scores by group and gender-row 1-2')
ax2.legend()

#3 bar graphs
labels = ['G1', 'G2', 'G3', 'G4', 'G5']
men_means = [20, 34, 30, 35, 27]
women_means = [25, 32, 34, 20, 25]

df = pd.DataFrame({"scores": [20, 34, 30, 35, 27, 25, 32, 34, 20, 25],
                   "gender": ['men', 'men', 'men','men','men','women','women','women','women','women'],
                   "label": ['G1',  'G2', 'G3', 'G4', 'G5', 'G1', 'G2', 'G3', 'G4', 'G5']})

sns.barplot(x='label', y='scores', data=df, ax=ax3, hue='gender', palette=['darkblue', 'lightblue'])
ax3.set_ylabel('Scores')
ax3.set_title('Seaborn - Scores by group and gender row 2-1')
ax3.set_xticks(x, labels)
ax3.legend()

#4 bar graphs
labels = ['G1', 'G2', 'G3', 'G4', 'G5']
men_means = [20, 34, 30, 35, 27]
women_means = [25, 32, 34, 20, 25]
ax4.bar(x - width/2, men_means, width, label='Men')
ax4.bar(x + width/2, women_means, width, label='Women')

ax4.set_ylabel('Scores')
ax4.set_title('Scores by group and gender row 2-2')
ax4.set_xticks(x, labels)
ax4.legend()

# set the spacing between subplots
# plt.subplots_adjust(left=0.1,
#                     bottom=0.1, 
#                     right=0.9, 
#                     top=0.9, 
#                     wspace=0.4, 
#                     hspace=0.4)

plt.show()

