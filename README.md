# Stock Market Project

# Project Description:
A PySpark based code that utilizes, yfinance and sklearn in order to train a model to predict the next day high of a stock that is given by the user.

Python Libraries Used:
- Pandas: For data manipulation, primarily to create dataframes with the saved data (most importantly true highest value, predicted highest value, and the date)
- yfinance: Crucial to access stock market information, this is a part of the Yahoo Finacne API.
- sklearn: Essential library for machine learning.
- tensorflow: Additional support for building neural networks.
- matplotlib: Easily visualise the accuracy of our predictions, both in a day to day and full month overview form.
- numpy 
- os

Breakdown:

1. The model will ask for a stock ticker, after the user enters the ticker information, all of the data from a designated start to end time will be downloaded.
2. This information is then narrowed down to just the Price at Open, at the stock's Highest/Lowest, and the Volume of trades.
3. The model then 


Goals:

The goal of this project is not to act as a substitute for doing research before commiting to a stock, it is to act as a tool to help the user more easily recognize patterns that have occured in the price throughout time, although the predicted values may not be identical to the actual daily high, it is a good indicator of whether the expected value of the stock will increase or decrease.

Stretch Goals:
1. 


Team Members:
- [Alyssa Berridge](https://github.com/A-bearr)
- [Arseniy Borsukov](https://github.com/arzingy)
- [Kieran Nguyen](https://github.com/kieranto1204)
- [Jay Boon](https://github.com/JHBoon)
- [Gennadiy Farladansky](https://github.com/genasha4168)
