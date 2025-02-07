import pandas as pd
from intervaltree import IntervalTree
import logging, sys, os

# Add the src directory to the path
sys.path.append(os.path.abspath(''))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='../logs/ip_geolocation.log')

logging.info('****************************Logging started for IP Geolocation module****************************')
class IPGeolocation:
    """
    A class to map IP addresses to their respective countries using an interval tree.
    Attributes:
    -----------
    tree : IntervalTree
        An interval tree that stores IP ranges and their corresponding countries.
    Methods:
    --------
    __init__(df_ranges):
        Initializes the IPGeolocation instance with IP ranges and builds the interval tree.
    map_ips_to_countries(df_ips):
        Maps a DataFrame of IP addresses to their respective countries.
    """

    def __init__(self, df_ranges):
        logging.info("Initializing IPGeolocation with IP ranges.")
        self.tree = IntervalTree()
        for _, row in df_ranges.iterrows():
            self.tree[row['lower_ip']:row['upper_ip'] + 1] = row['country']
        logging.info(
            "Interval tree built successfully with %d ranges.", len(df_ranges))

    def map_ips_to_countries(self, df_ips):
        """
        Maps IP addresses to their corresponding countries.
        This function takes a DataFrame containing IP addresses and maps each IP
        address to its corresponding country using an IP-to-country mapping tree.
            df_ips (pd.DataFrame): A DataFrame containing IP addresses with at least
                                   the following columns:
                                   - 'ip': The IP address as a string.
                                   - 'company': The company associated with the IP.
                                   - 'action': The action associated with the IP.
            pd.DataFrame: A DataFrame with the original columns plus an additional
                          'country' column indicating the country corresponding to
                          each IP address
        """
        logging.info(
            "Mapping IPs to countries for %d IP addresses.", len(df_ips))
        df_ips['ip_int'] = df_ips['ip'].astype(int)
        df_ips['country'] = df_ips['ip_int'].apply(lambda ip: next(
            iter(self.tree[ip])).data if self.tree[ip] else None)
        matched_count = df_ips['country'].notnull().sum()
        logging.info(
            "Successfully matched %d IPs to countries.", matched_count)
        return df_ips[['ip', 'company', 'action', 'country']]
