import sys
import math
import numpy as np
import matplotlib.pyplot as plt

# 기하분포의 pmf 값 계산을 위한 함수 정의.
def geometric_distribution(p, x):
    return pow((1-p), (x-1)) * p

# 기하 실험을 수행하는 함수.
def geometric_experiment(head_prob, step):

    # numpy의 random패키지 중 geometric함수를 이용해 동전던지기에 대한 기하실험을 한다.
    # random_var_head에 들어가는 값은 step만큼의 크기인 배열. 각 성분은 첫번째로 앞면이 나올때까지의 시도횟수를 의미.
    geom_exp_result = np.random.geometric(head_prob, step)
    random_var_head = geom_exp_result

    
    # 앞에서 실행한 기하실험에서 첫 성공이 나오기 까지 가장 많이 시도한 횟수를 max_num에 저장한다.
    max_num = np.max(random_var_head)
   
    
    # 기댓값 계산을 위한 변수.
    exp_x = 0
    exp_x2 = 0
    x_bin = []  # 앞면이 나올때까지의 시행횟수를 저장할 배열.
    
    # 반복문을 돌린다. 범위는 1부터 첫 성공이 나오기까지 가장 많이시도한 횟수+1 만큼 반복문을 돌린다.
    for i in range(1, max_num+1):
        x_bin.append(np.count_nonzero(random_var_head==i))
        # print(x_bin) 를 하면 볼수있다.
    
    # for문에서 기대값을 구하기 위해 범위가 필요하다. x_bin의 크기를 size_xbin에 저장한다. (시그마 i=1 ~ size_bin 까지)    
    size_xbin = len(x_bin)
    
    # x의 기댓값과 x^2의 기대값을 구한다.
    for i in range(size_xbin):
        exp_x = exp_x + (i * x_bin[i-1] / step)
        exp_x2 = exp_x2 + (pow(i,2) * x_bin[i-1] / step)
    
    # 위 for문에서 계산한 기대값들을 이용해 분산값을 구한다.    
    exp_var = exp_x2 - pow(exp_x, 2)
    
    
    # p값이 주어졌을때 기하분포의 기대값, 분산값을 계산한다. 
    real_mean = 1 / head_prob
    real_var = (1 - head_prob) / pow(head_prob, 2)
    

    print('          Experiment   Geometric')
    print('Mean:     %.4f,      %.4f'     %(exp_x, real_mean))
    print('Variance: %.4f,      %.4f'     %(exp_var, real_var))

    
    # pmf의 계산값을 저장하기 위한 배열.
    pmf_real = []
    
    # 첫 성공까지의 시행횟수가 가장 많은 수만큼 까지의 pmf식을 계산하고 그 값을 pmf_real에 넣어준다.
    # 실제 수식값 * step을 해서 실험값의 수치와 자리수를 맞추어 준다.
    for i in range(max_num):
        pmf_real.append(geometric_distribution(head_prob, i+1)*step)
    
    
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
        print('usage: ./geo.py head_prob step')
        sys.exit(1)
        
    head_prob = float(sys.argv[1])
    step = int(sys.argv[2])
    
    geometric_experiment(head_prob=head_prob, step=step)


if __name__ == '__main__':
    main()