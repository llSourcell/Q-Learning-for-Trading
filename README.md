# Teach Machine to Trade

This repo has code for the post: [Teach Machine to Trade](https://shuaiw.github.io/2018/02/11/teach-machine-to-trade.html)

### Dependencies

Python 2.7. To install all the libraries, run `pip install -r requirements.txt`


### Table of content

* `agent.py`: a Deep Q learning agent
* `envs.py`: a simple 3-stock trading environment
* `model.py`: a multi-layer perceptron as the function approximator
* `utils.py`: some utility functions
* `run.py`: train/test logic
* `requirement.txt`: all dependencies
* `data/`: 3 csv files with IBM, MSFT, and QCOM stock prices from Jan 3rd, 2000 to Dec 27, 2017 (5629 days). The data was retrieved using [Alpha Vantage API](https://www.alphavantage.co/)


### How to run

**To train a Deep Q agent**, run `python run.py --mode train`. There are other parameters and I encourage you look at the `run.py` script. After training, a trained model as well as the portfolio value history at episode end would be saved to disk.

**To test the model performance**, run `python run.py --mode test --weights <trained_model>`, where `<trained_model>` points to the local model weights file. Test data portfolio value history at episode end would be saved to disk.
