# Environment
gsl install reference: https://www.cnblogs.com/YangyaoCHEN/p/8189290.html 
```
sudo ./configure --prefix=/usr/local
sudo make
sudo make install

export C_INCLUDE_PATH=$C_INCLUDE_PATH:/usr/local/include
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/usr/local/include
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
```

# Compilation
//gcc -o test_ransac_tdoa main.c -lm -lgsl -lgslcblas

gcc -o test_ransac_tdoa main.c -lm

# Runing
./test_ransac_locator
