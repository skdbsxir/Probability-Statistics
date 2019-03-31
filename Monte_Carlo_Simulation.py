
import numpy as np
import matplotlib.pyplot as plt  #ToDo 01.
import matplotlib.patches as mpatches

fig = plt.figure(figsize = (9, 2.5))    # plot을 표시할 Fig객체 생성.

for column in range(3):
    subplot = fig.add_subplot(1, 3, column+1)   # X x Y의 Z번째
    subplot.set_xlim([0, 1])    # X축의 범위를 [0, 1) 로 설정.
    subplot.set_ylim([0, 1])    # Y축의 범위를 [0, 1) 로 설정.
    
    circle = plt.Circle(xy=(0,0), radius=1, color='black', fill=False)
    subplot.add_artist(circle)

    in_x, in_y = [], []
    out_x, out_y = [], []
    steps = 10*pow(10, column+1)
    matches = 0
    
    for _ in range(steps):
        
        x = np.random.rand(1)  #ToDo 02.
        y = np.random.rand(1)  #ToDo 02.

        isin = (x*x + y*y) <= 1
        matches += isin
        
        if isin:
            in_x.append(x)
            in_y.append(y)
            plt.scatter(x, y, marker='o', c='blue', s=0.1)  #ToDo 03.
        else:
            out_x.append(x)
            out_y.append(y)
            plt.scatter(x, y, marker='x', c='red', s=0.1)  #ToDo 03.
            
    plt.title('steps: ' + str(steps))
    PI = matches / steps * 4
    
    print('[steps]  %d' % steps)
    print('matches : %d / %d' % (matches, steps))
    print('pi : %.10f' % np.pi)
    print('approximate pi: %.10f' %PI)
    print('difference: %.10f\n' % (abs(np.pi - PI)))
    
    red_patch = mpatches.Patch(color='red', label='Miss')
    blue_patch = mpatches.Patch(color='blue', label='Hit')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), handles=[red_patch, blue_patch] )
    plt.show()  # ToDo 05.
    
    