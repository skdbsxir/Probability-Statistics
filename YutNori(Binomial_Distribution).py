import sys
import numpy as np
import matplotlib.pyplot as plt
import math


# 이항분포 식 구현. 성공횟수가 시행횟수보다 많은경우 종료(예외처리), 그렇지 않은경우 P(X=k)에 대한 pmf값을 반환.
def binomial_dist(n, k, p):
    if k > n:
        print ('k can not be greater than n')
        sys.exit(1)
    else:    
        return math.factorial(n) / math.factorial(n-k) / math.factorial(k) * math.pow(p,k) * math.pow((1-p), (n-k))
  
    
# 윷놀이의 구현. (xxx)표시가 되어있는 면이 나온 경우를 성공한 경우로 한다.   
def yutNori(step, prob_head=0.5):
  
    
  x = np.random.binomial(4, prob_head, step)
   # numpy.random.binomial(n, p, size)를 이용해 이항난수배열을 x에 저장한다.
   # x에 들어가는 값은 크기가 step인 배열. 배열의 각 성분값은 0~4가 된다.
   
  do = sum(x == 3) / step           # 성공한 경우가 3번. 즉, 4개의 윷에서 XXX가 적힌 면이 3번 나온경우. x의 배열에서 성분이 3인 경우만 뽑아내어 step만큼 나누어 확률값을 구한다.
  gae = sum(x == 2) / step          # 성공한 경우가 2번. 4개의 윷에서 XXX가 적힌 면이 2번 나온경우. x의 배열에서 성분이 2인 경우만 뽑아내 step만큼 나누어 확률값을 구한다. 
  geol = sum(x == 1) / step         # 성공한 경우가 1번. 4개의 윷에서 XXX가 적힌 면이 1번 나온경우. x의 배열에서 성분이 1인 경우만 뽑아내 step만큼 나누어 확률값을 구한다.
  yut = sum(x == 0) / step          # 성공한 경우가 0번. 4개의 윷 전부가 뒷면 (아무것도 안적힌 면)이 나온 경우. x의 배열에서 성분이 0인 경우만 뽑아내 step만큼 나누어 확률값을 구한다.
  mo = sum(x == 4) / step           # 성공한 경우가 4번. 4개의 윷 모두가 XXX가 적힌 면이 나온경우. x의 배열에서 성분이 4인 경우만 뽑아내 step만큼 나누어 확률값을 구한다.
  
  outcome_exp = [mo, do, gae, geol, yut]    # 구한 결과값들을 1개의 배열로 저장한다.
  
   #위에서 구현한 이항분포 식을 이용해 각 case에 대한 수학적인 확률값을 구한다.
  do_truth = binomial_dist(4, 3, prob_head)
  gae_truth = binomial_dist(4, 2, prob_head)
  geol_truth = binomial_dist(4, 1, prob_head)
  yut_truth = binomial_dist(4, 0, prob_head)
  mo_truth = binomial_dist(4, 4, prob_head)
  
  outcome_tru = [mo_truth, do_truth, gae_truth, geol_truth, yut_truth]      # 구한 결과값들을 1개의 배열로 저장한다.
  
  
      # 모,도,개,걸,윷 변수는 모두 같은 표본공간인 X에서 나온 값들. 총 합이 1이 되어야 확률의 정의를 만족하므로, 그렇지 않다면 프로그램을 종료한다.
  sum_of_probability = mo + do + gae + geol + yut
  
  if sum_of_probability != 1:
    print ('Sum of probability is not one')
    sys.exit(1)

  print ('Probability mo: %f, %f' %(mo, mo_truth) + \
        '\nProbability do: %f, %f' %(do, do_truth) + \
        '\nProbability gae: %f, %f' %(gae, gae_truth) + \
        '\nProbability geol: %f, %f' %(geol, geol_truth) + \
        '\nProbability yut: %f, %f' %(yut, yut_truth))
  
  
      # 도표에 이름을 적기 위한 배열 생성.
  objects = ('mo', 'do', 'gae', 'geol', 'yut')
  objects_truth = ('mo', 'do', 'gae', 'geol', 'yut')
      
      # 정해진 위치에 도표를 그리고 (subplot이용), 도표의 속성을 부여한 후(bar 이용), 도표의 이름을 작성한다(title 이용).
  plt.subplot(1, 2, 1)
  plt.bar(objects, outcome_exp, color = 'b')
  plt.title('experiment')
  
  plt.subplot(1, 2, 2)
  plt.bar(objects_truth, outcome_tru, color = 'b')
  plt.title('truth')
  plt.show()    #위에서 생성한 도표를 화면에 띄워준다.
  

def main():
  if len(sys.argv) != 2:
    print ('usage: ./yut.py step')
    sys.exit(1)

  step = int(sys.argv[1])

  yutNori(step=step)
  
if __name__ == '__main__':
    main()