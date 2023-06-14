import numpy as np
print("po operacji **-1 na array otrzymujemy :")
arr = np.array([[1., 2.], [3., 4.]])
print(arr)
arrinv = np.linalg.inv(arr)
print(arrinv)

print("typ wyniku otrzymanego po operacji to ", type(arrinv), "\n \npo tej samej operacji na matrix otrzymujemy :")

matr = np.matrix([[1., 2.], [3., 4.]])
print(matr)
matrinv = np.linalg.inv(matr)
print(matrinv)
print("typ wyniku to ", type(matrinv))
print("różnica między array i matrix to fakt, że operacje na array są wykonywane \nna każdym elemencie osobno "
      "podczas gdy operacje na matrix są wykonywane jak prawidłowe operacje na macierzach")