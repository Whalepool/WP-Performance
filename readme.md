# Whalepool Trading Performance Chart 

A simple script which plots your trades over the charts

Right now i've only tested/coded it for BTCUSD - and only works for bitfinex  

Welcome anyone who wants to make sure it works for other pairs, maybe using argparse, and/or also code a bitmex / other exchange implementation


## To run
You must export some envionment variables:  

```shell
export PERFORMANCE_SCRIPT_PATH='/home/username/path/to/whalepool-performance/'
export PERFORMANCE_USERNAME='my_username'
export BFX_API_KEY='my-bfx-api-key'
export BFX_API_SECRET='my-bfx-api-secret'
```

There is a file, `exports.sh` So you can just run...

```
source export.sh
python bitfinex.py
```


For more info join [@whalepoolbtc](https://t.me/whalepoolbtc) on telegram   
