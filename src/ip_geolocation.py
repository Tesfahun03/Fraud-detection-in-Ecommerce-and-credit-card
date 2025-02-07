import pandas as pd
from intervaltree import IntervalTree
import logging
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.abspath(''))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='../logs/ip_geolocation.log')

logging.info(
    '****************************Logging started for IP Geolocation module****************************')


class IPGeolocation:
    def __init__(self, df_ranges):
        try:
            logging.info("Initializing IPGeolocation with IP ranges.")
            self.tree = IntervalTree()
            for _, row in df_ranges.iterrows():
                self.tree[row['lower_bound_ip_address']
                    :row['upper_bound_ip_address'] + 1] = row['country']
            logging.info(
                "Interval tree built successfully with %d ranges.", len(df_ranges))
        except Exception as e:
            logging.error("Error initializing IPGeolocation: %s", str(e))
            raise

    def map_ips_to_countries(self, df_ips):
        try:
            logging.info(
                "Mapping IPs to countries for %d IP addresses.", len(df_ips))
            df_ips['ip_int'] = df_ips['ip_address'].astype(int)
            df_ips['country'] = df_ips['ip_int'].apply(lambda ip: next(
                iter(self.tree[ip])).data if self.tree[ip] else None)
            matched_count = df_ips['country'].notnull().sum()
            logging.info(
                "Successfully matched %d IPs to countries.", matched_count)
            return df_ips[['user_id', 'signup_time', 'purchase_time', 'purchase_value', 'device_id', 'source', 'browser', 'sex', 'age', 'ip_address', 'country', 'class']]
        except Exception as e:
            logging.error("Error mapping IPs to countries: %s", str(e))
            raise
