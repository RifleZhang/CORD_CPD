# Correlation-aware Change-point Detection <br> via Graph Neural Networks 
## Correlation-aware Dynamics Change-point Detection Model (CORD-CPD)
Paper link: https://arxiv.org/pdf/2004.11934.pdf

CORD_CPD aims to detect abrupt changes in multivariate time series, including:  
**Correlation Change**: the change of correlation structure of variables in multivariate time series  
**Independent Change**: the change of variable dynamics (physical property of variable's modus operandi)
![Intro Image](../master/gitImage/example.png)

## Simulation of Change-point Data
The first step to generate change-point data with scripts in folder data/:
```
cd data
./generate_change_point.sh
cd ..
```
Display the trajectory of varaibles and the red dot is the change-point (location and velocity changes are independent changes, connection changes is a correlation change): 
### independent change
![loc change](../master/gitImage/syn_loc.png)
![vel change](../master/gitImage/syn_vel.png)
### correlation change
![edge change](../master/gitImage/syn_edge.png)

## Training and Evaluation
To run/evaluate CORD-CPD, using following command:
```
./cmd.sh $GPU train
./cmd.sh $GPU test
```

Resulting detection of the three datasets
![results](../master/gitImage/result_plotting.png)

## How to use and next steps:
A motivating example (in our paper) is for financial markets: traders use pair trading strategy to profit from correlated stocks, such as Apple and Samsung (both are phone sellers), which share similar dips and highs. News about Apple expanding markets may independently raise its price without breaking its correlation with Samsung. However, news about Apple building self-driving cars will break its correlation with Samsung, and establish new correlations with automobile companies. While both of them are change-points, the former is an independent change of variables and the latter is a correlation change between variables. Knowing the type of change can guide financial experts to choose trading strategies properly.  

If interested, one can apply it on stock tickers, sensors or other related time series.

## Related Reference:

Although the code structure is different, some of the tools are from
https://github.com/ethanfetaya/NRI
