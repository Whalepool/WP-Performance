#!/usr/bin/env python3.6
from bitfinex import PublicClient, AuthClient1, AuthClient2
import datetime
import pandas as pd
import numpy as np
from pprint import pprint
import requests
import json
import time
import sys
import os
import logging

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from PIL import Image



# Configure Logging
FORMAT = '%(asctime)s -- %(levelname)s -- %(module)s %(lineno)d -- %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger('root')
logger.info('Beggining BFX performance script')


######################################
# Helper Functions 

# Checks for envionment variables being set
def check_envs(envars):
    # Check ENV variables
    errors = 0 
    for v in envars:
        if os.environ.get(v) is not None:
            pass
        else:
            errors += 1 
            logger.info('Please set a '+v+' envionment variable.')
    if errors > 0:
        sys.exit()


# Turns our trades into a dataframe 
def res_to_df(trades):
	df = pd.DataFrame(trades, dtype=float)

	# Set timestamp to datetime
	df['date'] = pd.to_datetime( df['timestamp'], unit='s' )
	df.set_index(df['date'], inplace=True)
	df.sort_index(inplace=True)

	# Deleted Unwanted rowes
	df.drop(['fee_amount','fee_currency','order_id', 'tid', 'timestamp', 'date'], axis=1, inplace=True)
	logger.info('Added '+str(len(df))+' trades to a dataframe')

	return df

# Turns our candles into a dataframe
def make_candle_df(results):
	df = pd.read_json(json.dumps(results))
	df.rename(columns={0:'date', 1:'open', 2:'close', 3:'high', 4:'low', 5:'volume'}, inplace=True)
	df['date'] = pd.to_datetime( df['date'], unit='ms' )
	df.set_index(df['date'], inplace=True)
	del df['date']
	df.sort_index(inplace=True)
	return df

# Makes our candlesticks for Matplotlib
def fooCandlestick(ax, quotes, width=0.03, colorup='#FFA500', colordown='#222', alpha=1.0):
	OFFSET = width/2.0
	lines = []
	boxes = []

	for q in quotes:

		timestamp, op, hi, lo, close = q[:5]
		box_h = max(op, close)
		box_l = min(op, close)
		height = box_h - box_l

		if close>=op:
			color = '#3fd624'
		else:
			color = '#e83e2c'

		vline_lo = Line2D( xdata=(timestamp, timestamp), ydata=(lo, box_l), color = 'k', linewidth=0.5, antialiased=True, zorder=10 )
		vline_hi = Line2D( xdata=(timestamp, timestamp), ydata=(box_h, hi), color = 'k', linewidth=0.5, antialiased=True, zorder=10 )
		rect = Rectangle( xy = (timestamp-OFFSET, box_l), width = width, height = height, facecolor = color, edgecolor = color, zorder=10)
		rect.set_alpha(alpha)
		lines.append(vline_lo)
		lines.append(vline_hi)
		boxes.append(rect)
		ax.add_line(vline_lo)
		ax.add_line(vline_hi)
		ax.add_patch(rect)

	ax.autoscale_view()

	return lines, boxes



######################################
# CONFIG

check_envs(['PERFORMANCE_SCRIPT_PATH','PERFORMANCE_USERNAME', 'BFX_API_KEY', 'BFX_API_SECRET'])
SCRIPT_PATH    = os.environ.get('PERFORMANCE_SCRIPT_PATH')
USERNAME       = os.environ.get('PERFORMANCE_USERNAME')
BFX_API_KEY    = os.environ.get('BFX_API_KEY')
BFX_API_SECRET = os.environ.get('BFX_API_SECRET')


# Lookback period
EPOCH                       = datetime.datetime.utcfromtimestamp(0)
TRADES_LOOKBACK_DAYS        = 3
TRADES_LOOKBACK_DATETIME    = datetime.datetime.now() - datetime.timedelta(days=TRADES_LOOKBACK_DAYS)
TRADES_LOOKBACK_MILISECONDS = str(int( (TRADES_LOOKBACK_DATETIME - EPOCH ).total_seconds() ))


# Chart settings 
CANDLE_STICK_TIMEFRAME   = '15m'
CANDLESTICK_WIDTHS       = { '1h'  : 0.3, '15m' : 0.01  }


# Initialize BFX Api Clients
client_v1 = AuthClient1( BFX_API_KEY, BFX_API_SECRET )
client_v2 = AuthClient2( BFX_API_KEY, BFX_API_SECRET )





######################################
# Get Trade History
logger.info('Requesting trades since: '+str(TRADES_LOOKBACK_DATETIME)+', in ms: '+str(TRADES_LOOKBACK_MILISECONDS))

# Fetch the first 200 trades
dfs = []
trades = client_v1.past_trades( params={ 'limit_trades': 200, 'timestamp': str(TRADES_LOOKBACK_MILISECONDS)   } )
dfs.append(res_to_df(trades))


# Get even more trades 
current_oldest_trade_ms = 0
while len(trades) > 0:

	# Get the MS of the oldest trade we have so far 
	current_oldest_trade_ms = float(trades[len(trades)-1]['timestamp'])-1

	logger.info('Requesting trades up until: '+str(current_oldest_trade_ms))

	# Fetch max 200 trades up until that point 
	trades = client_v1.past_trades( params={ 'limit_trades': 200, 'until': str(current_oldest_trade_ms), 'timestamp': str(TRADES_LOOKBACK_MILISECONDS)   } )

	# Check returned value
	if isinstance(trades, dict):

		# Check for API error, slow things down...
		if trades.get('error', False):
			logger.info('API Error: '+str(trades))
			logger.info('Sleeping for 20 seconds...')
			time.sleep(20)

	else:
		# No API error 
		if len(trades) > 0:

			# Set the oldest to be the last trade we just got 
			oldest = float(trades[len(trades)-1]['timestamp'])-1

			# Add to our trades dataframes list
			dfs.append(res_to_df(trades))


# Group the trades together 
bigdata = pd.concat(dfs)
bigdata.sort_index(inplace=True)

first_trade = bigdata.index[0].to_pydatetime()
last_trade = bigdata.index[-1].to_pydatetime()
total_trades = len(bigdata)

# Seperate into the buys and sells
# Buys
mask = (bigdata['type'] == 'Buy')
df_buy = bigdata[mask]
df_buy = df_buy.resample('15T').agg({'amount': 'sum', 'price': 'mean' })
df_buy = df_buy[pd.notnull(df_buy['amount'])]
df_buy = df_buy.reset_index()[['date','amount','price']]
df_buy['date'] = df_buy['date'].map(mdates.date2num)

# Sells 
mask = (bigdata['type'] == 'Sell')
df_sell = bigdata[mask]
df_sell = df_sell.resample('15T').agg({'amount': 'sum', 'price': 'mean' })
df_sell = df_sell[pd.notnull(df_sell['amount'])]
df_sell = df_sell.reset_index()[['date','amount','price']]
df_sell['date'] = df_sell['date'].map(mdates.date2num)


# pprint(df_buy.iloc[ [0, -1] ])
logger.info('-- Your Buy Trades, '+str(len(df_buy))+' in total --')
pprint(df_buy)

# pprint(df_sell.iloc[ [0, -1] ])
logger.info('-- Your Sell Trades, '+str(len(df_buy))+', in total --')
pprint(df_sell)



######################################
# Get Candles
current_ts          = datetime.datetime.now()
earliest_trade_date = (bigdata.iloc[-1].name).to_pydatetime()  + datetime.timedelta(days=TRADES_LOOKBACK_DAYS)
last_trade_date     = (bigdata.iloc[0].name).to_pydatetime()  - datetime.timedelta(days=TRADES_LOOKBACK_DAYS) 

earliest_trade_ms   = str( int( (current_ts - EPOCH).total_seconds() ) * 1000 )
last_trade_ms       = str( int( (earliest_trade_date - EPOCH).total_seconds() ) * 1000 )


candle_dfs = []
url = 'https://api.bitfinex.com/v2/candles/trade:'+CANDLE_STICK_TIMEFRAME+':tBTCUSD/hist?limit=%s&end=%s'
query = url % (200, earliest_trade_ms)
results = json.loads(requests.get(query).text)

while  isinstance(results, dict):
	logger.info('API Error: '+str(results))
	logger.info('Sleeping for 20 seconds...')
	time.sleep(20)
	results = json.loads(requests.get(query).text)

next_start_lookback = results[-1][0]
candle_dfs.append(make_candle_df(results))

while (int(next_start_lookback) > int(last_trade_ms)):
	query = url % (200, int(next_start_lookback)-1)
	pprint('Executing: '+query)
	results = json.loads(requests.get(query).text)
	if isinstance(results, dict):
		if results.get('error', False):
			pprint('API Error: '+str(results))
			pprint('Sleeping for 20 seconds...')
			time.sleep(20)
		else:
			pprint('Uknown dict returned: '+str(results))
			sys.exit()
	else:
		pprint('is a list...')
		if len(results) > 0:
			candle_dfs.append(make_candle_df(results))
			next_start_lookback = results[-1][0]

# Group the candles 
bigcandles = pd.concat(candle_dfs)
bigcandles.sort_index(inplace=True)
bigcandles = bigcandles.reset_index()[['date','open','high','low','close','volume']]

# Filter to just the ones we need 
mask = (bigcandles['date'] <= earliest_trade_date) & (bigcandles['date'] >= last_trade_date)
bigcandles = bigcandles[mask]
bigcandles['date'] = bigcandles['date'].map(mdates.date2num)




# Plot chart from csv
# Enable a Grid
plt.rc('axes', grid=True)
# Set Grid preferences 
plt.rc('grid', color='0.75', linestyle='-', linewidth=0.5)

# Create a figure, 16 inches by 12 inches
fig = plt.figure(facecolor='white', figsize=(16, 10), dpi=120)

# Draw 3 rectangles
# left, bottom, width, height
rect_chart = [0.05, 0.05, 0.75, 0.9]
rect_info = [0.8, 0.05, 0.15, 0.9]
# rect2 = [left, 0.27, width, 0.17]

ax1 = fig.add_axes(rect_chart, facecolor='#f6f6f6')  
ax1.set_xlabel('date')
ax1.xaxis_date()
ax1.set_title("My Paper Trades")
fooCandlestick(ax1, bigcandles.values, width=CANDLESTICK_WIDTHS[CANDLE_STICK_TIMEFRAME], colorup='g', colordown='k',alpha=0.4)
ax1.scatter(df_buy['date'], df_buy['price'], s=df_buy['amount']*3, c="g", zorder=11)
ax1.scatter(df_sell['date'], df_sell['price'], s=df_sell['amount']*3, c="r", zorder=11)


ax2 = fig.add_axes(rect_info, facecolor='#f6f6f6')  
ax2.plot(np.linspace(0.0, 2.0), np.linspace(0.0, 100.0), 'ko-', alpha=0)
ax2.set_yticklabels([])
ax2.set_xticklabels([])
ax2.grid(b=False)


 # Add the logo
im = Image.open(SCRIPT_PATH+'media/wp_logo.jpg')	
fig.figimage(   im,   105,  (fig.bbox.ymax - im.size[1])-29)

size = fig.get_size_inches()*fig.dpi
figure_width = size[0]
figure_height = size[1]

# Right box 
rb = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
rb_width = rb.width * fig.dpi
rb_width += 20
rb_height = rb.height * fig.dpi


ax2.text(1, 95, USERNAME, fontsize=14, horizontalalignment='center',fontweight='bold')

ax2.axhline(90, color='k', alpha=0.5)

ax2.text(1, 65, "Start:", fontsize=8, horizontalalignment='center')
ax2.text(1, 63, first_trade.strftime('%d %B \'%y'), fontsize=12, horizontalalignment='center')

ax2.text(1, 55, "End:", fontsize=8, horizontalalignment='center')
ax2.text(1, 53, last_trade.strftime('%d %B \'%y'), fontsize=12, horizontalalignment='center')

ax2.text(1, 45, "Duration:", fontsize=8, horizontalalignment='center')
ax2.text(1, 43, str((last_trade - first_trade).days)+" days", fontsize=12, horizontalalignment='center')

ax2.text(1, 35, "Total:", fontsize=8, horizontalalignment='center')
ax2.text(1, 33, total_trades, fontsize=12, horizontalalignment='center')


first_trade = bigdata.index[0].to_pydatetime()
last_trade = bigdata.index[-1].to_pydatetime()
total_trades = len(bigdata)


plt.savefig("performance.png")

# Clear the plot...
plt.clf()


logger.info('chart saved')