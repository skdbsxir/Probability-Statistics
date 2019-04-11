import sys
import numpy as np
import matplotlib.pyplot as plt
import math

    # X = 0~3 까지 임의로 1개를 생성한다. 각 숫자가 나올 확률은 0.2, 0.4, 0.3, 0.1 이다.
def generate_X(step):
    list_x = np.random.choice(4,step, [0.2, 0.4, 0.3, 0.1])

    return list_x

    # Y = 0~3 까지 임의로 1개를 생성한다. 각 숫자가 나올 확률은 0.25로 동일하다.
def generate_Y(step):
    list_y = np.random.choice(4, step, 0.25)
        
    return list_y
    
    # 생성한 숫자들을 가지고 작업을 하는 함수.
def LetsDoIt(step):
    x = generate_X(step)
    y = generate_Y(step)
    
    # 그래프에 mapping될 2차원 numpy배열을 생성한다.
    # numpy.zeros를 통해 전부 0으로 초기화한다.
    result = np.zeros((4,4))
    
    # 0~3이 나온 횟수를 count해서 result 배열에 저장한다.
    # 배열위치에 해당되는 값(0~3)이 나오면, 값을 +1씩 해준다.
    # 최종 결과는 4x4 크기의 2차원 배열에 1~4가 몇번씩 나왔는지 출력이 된다.    
    for i in range(step):
        result[x[i], y[i]] += 1
    
    # 생성된 result 배열을 이용해 각 원소의 출현확률을 계산하여, scatter 그래프로 그려준다.
    for i in range(0, 4):
        for j in range(0, 4):
            plt.scatter(x+1, y+1, c='b')
            plt.text(i+1.1,j+1.1, "{}".format(float(result[i,j] / step)), fontsize=10)  # 각 점에 해당되는 확률값을 기록한다. 확률값은 '나온횟수 / 총 시행횟수' 가 된다.
            
    # np.sum(array, axis) : 전체 값을 계산하고자 하는 배열과 계산할 축을 넣는다.
    # axis = 0 : 그래프상의 y축 //  1 : x축 
    x_axis_sum = np.sum(result, 1)  # x축의 성분을 전부 더한다. return값은 크기 4의 list이며, 각 list 성분은 x=1,2,3,4에 대응되는 합산값이다.
    y_axis_sum = np.sum(result, 0)  # y축의 성분을 전부 더한다. return값은 크기 4의 list이며, 각 list 성분은 y=1,2,3,4에 대응되는 합산값이다.
    print('\n-------------<Problem 3>--------------')
    print('\nPx(4) is %0.2f' %(x_axis_sum[3] / step))   
    print('Py(1) is %0.2f' %(y_axis_sum[0] / step))
    
    # result배열의 (2,3)위치 (배열에선 (1,2) 위치)의 값을 y=3일때의 sample space로 축약해서 조건부 확률값을 계산한다.
    print('Px|y(2|3) is %0.2f\n' %(result[1,2]/y_axis_sum[2]))
    
    
    print('-------------<Problem 4>--------------')
    # x값들의 기댓값을 구한다.
    # mu_x는 numpy.random에서 구한 값들로 계산한 실험값이고, exp_x는 numpy.mean method를 이용한 이론값이다.
    mu_x = (1*(x_axis_sum[0] / step)) + (2*(x_axis_sum[1] / step)) + (3*(x_axis_sum[2] / step)) + (4*(x_axis_sum[3] / step))
    exp_x = np.mean([1, 1, 2, 2, 2, 2, 3, 3, 3, 4])
    print('\nX`s computed mean : %0.02f \nX`s method-calculated mean : %0.02f\n' %(mu_x, exp_x))
    
    
    # y값들의 기댓값을 구한다.
    # 마찬가지로, mu_y는 numpy.random에서 구한 값들로 계산한 실험값이고, exp_y는 numpy.mean을 이용한 이론값이다.
    mu_y = (1*(y_axis_sum[0] / step)) + (2*(y_axis_sum[1] / step)) + (3*(y_axis_sum[2] / step)) + (4*(y_axis_sum[3] / step))
    exp_y = np.mean([1, 2, 3, 4])
    print('Y`s computed mean : %0.02f \nY`s method-calculated mean : %0.02f\n' %(mu_y, exp_y))
    
    
    print('-------------<Problem 5>--------------')
    # x값들의 분산값을 구한다.
    # sigma_x는 실제 X의 2차적률을 구하고 그 값에 (E[X])^2 를 빼준 값이고, var_x는 numpy.var method를 이용한 이론값이다.
    sigma_x = ((1*(x_axis_sum[0] / step)) + (4*(x_axis_sum[1] / step)) + (9*(x_axis_sum[2] / step)) + (16*(x_axis_sum[3] / step))) - pow(mu_x, 2)
    var_x = np.var([1, 1, 2, 2, 2, 2, 3, 3, 3, 4]) 
    print('\nX`s computed variance : %0.02f \nX`s method-calculated varicance : %0.02f\n' %(sigma_x, var_x))
    
    
    # y값들의 분산값을 구한다.
    # 마찬가지로, sigma_y는 실제 Y의 2차적률을 구하고 그 값에 (E[X])^2 를 빼준 값이고, var_y는 numpy.var를 이용한 이론값이다.
    sigma_y = ((1*(y_axis_sum[0] / step)) + (4*(y_axis_sum[1] / step)) + (9*(y_axis_sum[2] / step)) + (16*(y_axis_sum[3] / step))) - pow(mu_y, 2)
    var_y = np.var([1, 2, 3, 4]) 
    print('Y`s computed variance : %0.02f \nY`s method-calculated varicance : %0.02f\n' %(sigma_y, var_y)) 
    
    
    # 주어진 r.v의 선형결합 형태를 선형성의 원리를 적용해 값을 구한다.
    # E[aX + b] = a*E[X] + b // Var(aX+b) = a^2 * Var(X)
    print('-------------<Problem 6>--------------')
    print('\nE[2X+4]`s value : %0.02f' %((2*mu_x) + 4))
    print('E[-Y^2 - 1]`s value : %0.02f' %((-pow(mu_y, 2)) - 1))
    print('Var[3Y-3]`s value : %0.02f\n' %(9*sigma_y))
    
    
    # X와 Y의 결합분포의 확률 Px,y(X,Y)와 X와 Y의 한계분포 확률의 곱 Px(X)*Py(Y)의 값이 같은지, 틀린지를 비교한다.
    # 두 값이 같다면, X와 Y는 서로 statistically independent할 것이고
    # 만약 다르다면 X와 Y는 서로 dependent할 것이다.
    print('-------------<Problem 7>--------------')
    px4 = x_axis_sum[3] / step
    py1 = y_axis_sum[0] / step
    pxy41 = result[3, 0] / step
    print('\nPx(4) : %f' %(px4) + \
          '\nPy(1) : %f' %(py1) + \
          '\nPx,y(4,1) : %f' %(pxy41) +\
          '\nPx(4) * Py(1) : %f' %(px4*py1) + \
          '\n\n%f is not equal to %f' %(px4*py1, pxy41) +\
          '\n\nTherefore, X and Y are dependent,\n')
    # 확인한 결과, 결합분포의 확률값과 X와 Y의 한계분포 확률의 곱이 다르게 나왔으며, 이는 X와 Y가 서로 dependent하다는 의미이다.
    
    '''
    # 다른 방법으로, r.v X와 Y의 공분산 값을 구해서 그 값이 0인지 아닌지를 확인한다.
    # 공분산값이 0이면 둘은 서로 독립, 0이 아니라면 X와 Y는 dependent할 것이고 결합분포 확률은 각각의 한계분포 확률의 곱으로 나타낼 수 없게된다.
    print('Or, just calculate covariance of X,Y')
    print('\nIf Cov[X,Y] = E[(X-E[X])(Y-E[Y])] is equal to 0, X and Y are independent. If not, X and Y are dependent.')
    
    # 공분산 정의식 Cov[X,Y] = E[(X-E[X])(Y-E[Y])] = (sigma_x)(sigma_y)[(x-E[X])(x-E[Y])*Px,y(x,y)]에서 E[X],E[Y]엔 numpy.random에서 구한 기대값을 대입한다.
    cov_xy = ((1-mu_x)*(1-mu_y)*(result[0, 0]/step)) + ((2-mu_x)*(2-mu_y)*(result[1, 1]/step)) + ((3-mu_x)*(3-mu_y)*(result[2, 2]/step)) + ((4-mu_x)*(4-mu_y)*(result[3, 3]/step))
    print('\nCov[X,Y] : %f' %(cov_xy))
    print('\nCov[X,Y] is not 0. So, X and Y are dependent.\n')
    '''
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