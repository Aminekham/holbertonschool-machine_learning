#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
persons = ['Farrah', 'Fred', 'Felicia']
fruits = ['Apples', 'Bananas', 'Oranges', 'Peaches']
fig, ax = plt.subplots()

for i in range(len(fruits)):
    ax.bar(persons, fruit[i], label=fruits[i], color=colors[i], bottom=np.sum(fruit[:i], axis=0), width=[0.5])

ax.set_ylabel('Quantity of Fruit')
ax.set_title('Number of Fruit per Person')
ax.set_ylim(0, 80)
ax.set_yticks(np.arange(0, 81, 10))
ax.legend()

plt.show()
