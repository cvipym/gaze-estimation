import csv
import numpy as np
def save_weights(name='HSK'):
    x,y,xi_hat,yi_hat,yl = [] ,[] ,[] ,[] ,[]
    with open('result.csv', 'r') as file:
        # CSV 파일을 읽는 CSV 리더 객체 생성
        csv_reader = csv.reader(file)
        
        # CSV 데이터를 저장할 빈 리스트 생성
        data = []
        
        # CSV 파일의 각 행을 읽어서 리스트에 추가
        for row in csv_reader:
            x.append(row[0])
            y.append(row[1])
            xi_hat.append(row[2])
            yi_hat.append(row[3])
            yl.append(row[4])
            
        
    # NumPy 배열로 변환
    x = np.array(x,dtype='float64')
    y = np.array(y,dtype='float64')
    xi_hat = np.array(xi_hat,dtype='float64')
    yi_hat = np.array(yi_hat,dtype='float64')
    yl = np.array(yl,dtype='float64')


    X = np.column_stack((yi_hat, yl, np.ones_like(yi_hat)))
    weights = np.linalg.lstsq(X, y-yi_hat, rcond=None)[0]

    # 결과 출력
    w1, w2, w3 = weights
    np.save(f'data/finetune/{name}',np.array([w1,w2,w3]))
    print("w1:", w1)
    print("w2:", w2)
    print("w3:", w3)


