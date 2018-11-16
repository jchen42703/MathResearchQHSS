import datetime
import quandl
import pandas as pd
import numpy as np 
from scipy import stats 

quandl.ApiConfig.api_key = 'YwMn-jZt3hjv1AXZS57Z'

class EventLoader_Custom(object):
  '''
  Loads data for adjusted closing stock prices on a specified window around a date
  Attributes:
    date_string: YYYYMMDD
    window: int
    tickers: list of [company, reference] symbols
  '''
  def __init__(self, date_string, window = 15, ticker = ['AAPL', 'MSFT']):
      self.date_string = date_string
      self.window = window
      self.ticker = ticker
    
  def data_load(self):
      '''
      loading sorted data table around specified date and window
      '''
      datetime_object = datetime.datetime.strptime(self.date_string, '%Y%m%d')
      gte = datetime_object - datetime.timedelta(self.window)
      lte = datetime_object + datetime.timedelta(self.window)
      # get the table for daily stock prices and,
      # filter the table for selected tickers, columns within a time range
      # set paginate to True because Quandl limits tables API to 10,000 rows per call
      
      data = quandl.get_table('WIKI/PRICES', ticker = self.ticker[0], 
                              qopts = { 'columns': ['ticker', 'date', 'adj_close'] }, 
                              date = { 'gte': gte, 'lte': lte }, 
                              paginate=True) 
      #
      sorted_df = data.sort_values(by='date')
      new = sorted_df.set_index('date').drop(['ticker'], axis = 1)
      return new
  
  def load_reference(self):
    '''
    loading reference market data with window and time
    '''
    data = quandl.get_table('WIKI/PRICES', ticker = self.ticker[1], 
                        qopts = { 'columns': ['ticker', 'date', 'adj_close'] }, 
                        date = { 'gte': gte, 'lte': lte }, 
                        paginate=True) 
                        
class EventStudy(object):
  '''
    Produces metrics for event study.
    Attributes:
      data: pandas datafrane of adjusted stock prices for company around day of event
      market: pandas dataframe of adjusted stock prices of reference market around the day of event
  '''
  def __init__(self, data, market):
    self.data = data
    self.market = market

  @staticmethod
  def returns(data, basedOn=1, cc=False, col=None):
      '''
      Computes stepwise returns; usually daily
      Parameters
      ----------
          data: numpy.array or pandas.Series or pandas.DataFrame
          cc: boolean, if want the continuously compounded return
          basedOn: Calculate the returns basedOn the previously n entries
              For example if the data is monthly and basedOn=12 is an Annual Return
          col=None: if data is pandas.DataFrame use this column to calculate the Daily Returns
      Returns
      -------
          if data is numpy.array of 1 dim: numpy.array with the daily returns
          if data is numpy.array of 2 dim: numpy.array with the daily returns of each column
          if data is pandas.Series: pandas.Series with the daily returns
          if data is pandas.DataFrame and col is None: pandas.DataFrame with the daily returns of each column
          if data is pandas.DataFrame and col is not None: pandas.Series with the daily returns
      '''

      if type(data) is np.ndarray or type(data) is list:
          dr = np.zeros(shape=data.shape)
          if cc:
              # return np.log(data[basedOn:] / data[0:-basedOn])
              dr[basedOn:] = np.log(data[basedOn:] / data[0:-basedOn])
              return dr
          else:
              # return data[basedOn:] / data[0:-basedOn] - 1
              dr[basedOn:] = data[basedOn:] / data[0:-basedOn] - 1
              return dr

      if type(data) is pd.Series or type(data) is pd.Series:
          ans = EventStudy.returns(data.values, cc=cc,  basedOn=basedOn)
          name = data.name
          if cc:
              name = name + ' CC'
          name = name + ' returns'
          if basedOn != 1:
              name = name + ' (' + str(basedOn) + ')'
          return pd.Series(ans, index=data.index, name=name)

      if type(data) is pd.DataFrame:
          if col is not None:
              return EventStudy.returns(data[col], cc=cc,  basedOn=basedOn)
          else:
              return data.apply(EventStudy.returns, cc=cc, basedOn=basedOn)
  
  def market_return(self):
    '''
    Sets attributes for the Event Study class for metrics.
    '''
    # 1. Linear Regression: On the estimation_period
    dr_data = EventStudy.returns(self.data)
    dr_market = EventStudy.returns(self.market)
    
    c_name = dr_data.columns[0]
    x =  dr_market[c_name]
    y = dr_data[c_name]
    assert x.shape[0] > 0
    print(x.shape)
    slope, intercept, r_value, p_value, std_error = stats.linregress(x, y)
    er = lambda x: x * slope + intercept

    # 2. Analysis on the event window
    # Expexted Return:
    self.er = dr_market.apply(er)[c_name]
    self.er.name = 'Expected return'
    # Abnormal return: Return of the data - expected return
    self.ar = dr_data[c_name] - self.er
    self.ar.name = 'Abnormal return'
    # Cumulative abnormal return
    self.car = self.ar.cumsum()
    self.car.name = 'Cum abnormal return'
    # t-test
    t_test_calc = lambda x: x / std_error
    self.t_test = self.ar.apply(t_test_calc)
    self.t_test.name = 't-test'
    self.prob = self.t_test.apply(stats.norm.cdf)
    self.prob.name = 'Probability'

  def summary(self):
      '''
      Produces a dataframe summary of all metrics. 
      * Used after self.market_return()
      '''
      metrics_dict = {'Expected Return': self.er, 'Abnormal Return': self.ar,
                      'Cumulative Abnormal Return': self.car, 'T-Test': self.t_test,
                      'p-value': self.prob
                      }
      return pd.DataFrame.from_dict(metrics_dict, orient = 'index', columns = ['Metric Values'])
