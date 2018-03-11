import numpy as np
import os
import pandas as pd
import random
import time
import logging
from sklearn.preprocessing import MinMaxScaler

random.seed(time.time())


class StockDataSet(object):
    def __init__(self,
                 stock_sym,
                 input_size=1,
                 num_steps=30,
                 test_ratio=0.1,
                 normalized=True,
                 close_price_only=True):
        self.stock_sym = stock_sym
        self.input_size = input_size
        self.num_steps = num_steps
        self.test_ratio = test_ratio
        self.close_price_only = close_price_only
        self.normalized = normalized

        # Read csv file
        self.raw_df = pd.read_csv(os.path.join("data", "%s.csv" % stock_sym))
        logging.info('self.raw_df.shape = {}'.format(self.raw_df.shape))
        self.raw_df = self.raw_df.iloc[::-1]
        logging.info('{}'.format(self.raw_df.head()))

        # Merge into one sequence
        if close_price_only:
            self.raw_seq = self.raw_df['Close'].tolist()
        else:
            self.raw_seq = [price for tup in self.raw_df[['Open', 'Close']].values for price in tup]

        self.raw_seq = np.array(self.raw_seq)
        self.train_X, self.train_y, self.test_X, self.test_y = self._prepare_data(self.raw_seq)

    def info(self):
        return "StockDataSet [%s] train: %d test: %d" % (
            self.stock_sym, len(self.train_X), len(self.test_y))

    # prices: [number of price window, price window]
    # return: normalized prices
    def _normalize_price_to_change_rate(self, prices):
        prices = [prices[0] / prices[0][0] - 1.0] + [
            curr / prices[i][-1] - 1.0 for i, curr in enumerate(prices[1:])]
        prices = np.array(prices)
        return prices

    def _normalize_price_to_range(self, prices, min = 0.0, max = 1.0):
        row_num = prices.shape[0]
        col_num = prices.shape[1]
        prices.reshape(row_num * col_num, 1)
        scaler = MinMaxScaler(feature_range=(min, max))
        self.px_scaler = scaler.fit(prices)
        print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
        logging.info(
            '{} {}'.format(scaler.data_min_, scaler.data_max_))
        # normalize the dataset and print the first 5 rows
        prices = self.px_scaler.transform(prices)
        prices.reshape(row_num, col_num)
        return prices

    def denormalize_px(self, normalized):
        result = self.px_scaler.inverse_transform(normalized)
        return result

    def _prepare_data(self, seq):
        # split into items of input_size
        seq = [np.array(seq[i * self.input_size: (i + 1) * self.input_size])
               for i in range(len(seq) // self.input_size)]
        # seq = num_days * [price]
        logging.info('len(seq) = {}, seq[0].shape = {}'.format(len(seq), seq[0].shape))

        if self.normalized:
            seq = self._normalize_price_to_change_rate(seq)
            seq = self._normalize_price_to_range(seq)

        # split into groups of num_steps
        X = np.array([seq[i: i + self.num_steps] for i in range(len(seq) - self.num_steps)])
        y = np.array([seq[i + self.num_steps] for i in range(len(seq) - self.num_steps)])
        # 3360

        train_size = int(len(X) * (1.0 - self.test_ratio))
        train_X, test_X = X[:train_size], X[train_size:]
        train_y, test_y = y[:train_size], y[train_size:]
        # train_X = num_days * num_steps (30) * input_size (1)
        # train_y = num_days * input_size (1)

        logging.info('train_size = {}'.format(train_size))
        logging.info('self.raw_df.shape = {}'.format(self.raw_df.shape))

        logging.info('start date {}'.format(
            self.raw_df.iloc[[train_size - 1 + self.num_steps,
                              train_size + self.num_steps,
                              -1]]))

        return train_X, train_y, test_X, test_y

    def generate_one_epoch(self, batch_size):
        num_batches = int(len(self.train_X)) // batch_size
        if batch_size * num_batches < len(self.train_X):
            num_batches += 1

        batch_indices = list(range(num_batches))
        # print(batch_indices)
        random.shuffle(batch_indices)
        for j in batch_indices:
            batch_X = self.train_X[j * batch_size: (j + 1) * batch_size]
            batch_y = self.train_y[j * batch_size: (j + 1) * batch_size]
            assert set(map(len, batch_X)) == {self.num_steps}
            yield batch_X, batch_y


