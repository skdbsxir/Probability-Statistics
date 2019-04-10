import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy import array


# np.random.geomtric(prob, step)을 구현한다.
def geom_randVar(step, prob):
    
    # 결과를 return할 list를 생성.
    success = []
    
    for i in range(step) :
        # 초기확률값과 시행횟수는 0으로 초기화.
        generated_prob = 0
        count_trial = 0
        
        # 난수로 생성된 확률값이 주어진 조건 하에 1인지 0인지를 구별. 1인경우, 이를 success list에 추가하고, while문을 빠져나온다.
        # 이 과정을 step번 만큼 실행. 최종 결과물은 step크기 만큼의 success list가 생성된다.
        while generated_prob != 1:
        
            # 몇번째 시도인지 count.
            count_trial += 1
            generated_prob = np.random.random()
            
            if generated_prob < prob :
                generated_prob = 0
            else :
                generated_prob = 1
                success.append(count_trial)
                
    return success
            
# 기하분포의 pmf 값 계산을 위한 함수 정의.
def geometric_distribution(p, x):
    return pow((1-p), (x-1)) * p


# 기하 실험을 수행하는 함수.
def geometric_experiment(step, prob):

    # numpy의 random패키지 중 geometric함수를 이용해 동전던지기에 대한 기하실험을 한다.
    # random_var_head에 들어가는 값은 step만큼의 크기인 배열. 각 성분은 첫번째로 앞면이 나올때까지의 시도횟수를 의미.
    # 우리는 현재 직접 구현한 geometric함수, geom_randVar를 사용할 것이다.
    geom_exp_result = geom_randVar(step, prob)
    
    # 반환된 list는 일반 list형. 이를 numpy형 array로 변환해준다.
    random_var_head = array(geom_exp_result)
    
    # 앞에서 실행한 기하실험에서 첫 성공이 나오기 까지 가장 많이 시도한 횟수를 max_num에 저장한다.
    max_num = np.max(random_var_head)
   
    # 기댓값 계산을 위한 변수.
    exp_x = 0
    exp_x2 = 0
    x_bin = []  # 앞면이 나올때까지의 시행횟수를 저장할 배열.

    # 반복문을 돌린다. 범위는 1부터 첫 성공이 나오기까지 가장 많이시도한 횟수+1 만큼 반복문을 돌린다.
    for i in range(1, max_num+1):
        # random_var_head에서 같은게 몇개 있는지 센 다음, x_bin에 차례러 넣어준다.
        x_bin.append(np.count_nonzero(random_var_head == i))
        
    # for문에서 기대값을 구하기 위해 범위가 필요하다. x_bin의 크기를 size_xbin에 저장한다. (시그마 i=1 ~ size_bin 까지)    
    size_xbin = len(x_bin)
    
    # x의 기댓값과 x^2의 기대값을 구한다.
    for i in range(size_xbin):
        exp_x = exp_x + (i * x_bin[i-1] / step)
        exp_x2 = exp_x2 + (pow(i,2) * x_bin[i-1] / step)
    
    # 위 for문에서 계산한 기대값들을 이용해 분산값을 구한다.    
    exp_var = exp_x2 - pow(exp_x, 2)
    
    
    # p값이 주어졌을때 기하분포의 기대값, 분산값을 계산한다. 
    real_mean = 1 / prob
    real_var = (1 - prob) / pow(prob, 2)
    

    print('          Experiment   Geometric')
    print('Mean:     %.4f,      %.4f'     %(exp_x, real_mean))
    print('Variance: %.4f,      %.4f'     %(exp_var, real_var))

    
    # pmf의 계산값을 저장하기 위한 배열.
    pmf_real = []
    
    # 첫 성공까지의 시행횟수가 가장 많은 수만큼 까지의 pmf식을 계산하고 그 값을 pmf_real에 넣어준다.
    # 실제 수식값 * step을 해서 실험값의 수치와 자리수를 맞추어 준다.
    for i in range(max_num):
        pmf_real.append(geometric_distribution(prob, i+1)*step)
    
    
    # 그래프의 아래부분의 간격은 max_sum까지 그리도록 한다.
    index = np.arange(max_num)
    
    plt.plot()
    # 각각 실험과 실제 계산값을 비교하기위해 그래프를 2개 그린다.
    plt.bar(index+0.00, x_bin, 0.3, label = 'Experiment',color='b')
    plt.bar(index+0.3, pmf_real, 0.3, label = 'Truth', color='g') 
    plt.legend()
    plt.title('geometric test')
    plt.show()
        

def main():
    if len(sys.argv) != 3:
        print('usage: ./geo.py step prob')
        sys.exit(1)
        
    prob = float(sys.argv[1])
    step = int(sys.argv[2])
    
    geometric_experiment(step = step, prob = prob)


if __name__ == '__main__':
    main()