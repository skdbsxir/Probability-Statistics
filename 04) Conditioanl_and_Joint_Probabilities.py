import sys
import numpy as np
import matplotlib.pyplot as plt
import math


def generate_X(step):
    # X = 1~4 까지. 나올 확률은 0.2, 0.4, 0.3, 0.1
    list_x = np.random.choice(4,step, [0.2, 0.4, 0.3, 0.1])

    return list_x

def generate_Y(step):
    # Y = 1~4 까지. 나올 확률은 0.25로 동일.
    list_y = np.random.choice(4, step, 0.25)
        
    return list_y
    
def LetsDoIt(step):
    x = generate_X(step)
    y = generate_Y(step)
    
    result = np.zeros((4,4))
    # 나온 횟수를 count해서 저장하자.    
    for i in range(step):
        result[x[i], y[i]] += 1
    
    for i in range(0, 4):
        for j in range(0, 4):
            plt.scatter(x+1, y+1, c='b')
            plt.text(i+1.1,j+1.1, "{}".format(float(result[i,j] / step)), fontsize=10)
    # np.sum(array, axis) : 계산하고자 하는 배열과 계산할 축을 넣는다.
    # axis = 0 : 그래프상의 y축 //  1 : x축 
    x_axis_sum = np.sum(result, 1)
    y_axis_sum = np.sum(result, 0)
    print('\n-------------<Problem 3>--------------')
    print('\nPx(4) is %0.2f' %(x_axis_sum[3] / step))
    print('Py(1) is %0.2f' %(y_axis_sum[0] / step))
    # result배열의 (2,3)위치 (배열에선 (1,2) 위치)의 값을 
    # y=3일때의 sample space로 축약해서 조건부 값을 계산한다.
    print('Px|y(2|3) is %0.2f\n' %(result[1,2]/y_axis_sum[2]))
    
    print('-------------<Problem 4>--------------')
    #x값들의 기댓값을 구한다.
    mu_x = (1*(x_axis_sum[0] / step)) + (2*(x_axis_sum[1] / step)) + (3*(x_axis_sum[2] / step)) + (4*(x_axis_sum[3] / step))
    exp_x = np.mean([1, 1, 2, 2, 2, 2, 3, 3, 3, 4])
    print('\nX`s computed mean : %0.02f \nX`s method-calculated mean : %0.02f\n' %(mu_x, exp_x))
    
    #y값들의 기댓값을 구한다.
    mu_y = (1*(y_axis_sum[0] / step)) + (2*(y_axis_sum[1] / step)) + (3*(y_axis_sum[2] / step)) + (4*(y_axis_sum[3] / step))
    exp_y = np.mean([1, 2, 3, 4])
    print('Y`s computed mean : %0.02f \nY`s method-calculated mean : %0.02f\n' %(mu_y, exp_y))
    
    print('-------------<Problem 5>--------------')
    #x값들의 분산값을 구한다.
    sigma_x = ((1*(x_axis_sum[0] / step)) + (4*(x_axis_sum[1] / step)) + (9*(x_axis_sum[2] / step)) + (16*(x_axis_sum[3] / step))) - pow(mu_x, 2)
    var_x = np.var([1, 1, 2, 2, 2, 2, 3, 3, 3, 4]) 
    print('\nX`s computed variance : %0.02f \nX`s method-calculated varicance : %0.02f\n' %(sigma_x, var_x))
    
    #y값들의 분산값을 구한다.
    sigma_y = ((1*(y_axis_sum[0] / step)) + (4*(y_axis_sum[1] / step)) + (9*(y_axis_sum[2] / step)) + (16*(y_axis_sum[3] / step))) - pow(mu_y, 2)
    var_y = np.var([1, 2, 3, 4]) 
    print('Y`s computed variance : %0.02f \nY`s method-calculated varicance : %0.02f\n' %(sigma_y, var_y)) 
    
    print('-------------<Problem 6>--------------')
    print('\nE[2X+4]`s value : %0.02f' %((2*mu_x) + 4))
    print('E[-Y^2 - 1]`s value : %0.02f' %((-pow(mu_y, 2)) - 1))
    print('Var[3Y-3]`s value : %0.02f\n' %(9*sigma_y))
    
    print('-------------<Problem 7>--------------')
    px4 = x_axis_sum[3] / step
    py1 = y_axis_sum[0] / step
    pxy41 = result[3, 0] / step
    print('\nPx(4) : %f' %(px4) + \
          '\nPy(1) : %f' %(py1) + \
          '\nPx,y(4,1) : %f' %(pxy41) +\
          '\nPx(4) * Py(1) : %f' %(px4*py1) + \
          '\n\nTherefore, X and Y are dependent')
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()
def main():
  if len(sys.argv) != 2:
    print ('usage: ./yut.py step')
    sys.exit(1)

  step = int(sys.argv[1])

  LetsDoIt(step=step)
  
if __name__ == '__main__':
    main()