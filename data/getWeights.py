import csv
import numpy as np
# import matplotlib.pyplot as plt

def save_weights(name='HSK'):
    x,y,xi,yi,yl = [] ,[] ,[] ,[] ,[]
    with open(f'data/finetune/{name}.csv', 'r') as file:
        # CSV 파일을 읽는 CSV 리더 객체 생성
        csv_reader = csv.reader(file)
        
        # CSV 데이터를 저장할 빈 리스트 생성
        data = []
        
        # CSV 파일의 각 행을 읽어서 리스트에 추가
        for row in csv_reader:
            x.append(row[0])
            y.append(row[1])
            xi.append(row[2])
            yi.append(row[3])
            yl.append(row[4])
            
        
    # NumPy 배열로 변환
    x = np.array(x,dtype='float64')
    y = np.array(y,dtype='float64')
    xi = np.array(xi,dtype='float64')
    yi = np.array(yi,dtype='float64')
    yl = np.array(yl,dtype='float64')


    
    X = np.column_stack((yi, yl, np.ones_like(yi)))
    weights = np.linalg.lstsq(X, y-yi, rcond=None)[0]

    # 결과 출력
    w1, w2, w3 = weights
    np.save(f'data/finetune/{name}',np.array([w1,w2,w3]))
    # print("w1:", w1)
    # print("w2:", w2)
    # print("w3:", w3)
    

if __name__ == '__main__':
    save_weights()

