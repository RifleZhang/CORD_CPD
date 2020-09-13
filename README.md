# CORD_CPD
Correlation-aware Change-point Detection via Graph Neural Networks

The first step to generate change-point data with scripts in folder data/:
```
cd data
./generate_change_point.sh
cd ..
```

To run/evaluate CORD-CPD, using following command:
```
./cmd.sh $GPU train
./cmd.sh $GPU test
```

Related Reference:
Although the code structure is different, some of the tools are from:
https://github.com/ethanfetaya/NRI
