# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


from ..utils import Singleton
from pymongo import MongoClient
from bson.objectid import ObjectId
from collections import OrderedDict
from datetime import datetime
from time import time
import pandas as pd


class DBClient(metaclass=Singleton):
    def __init__(self, *args, **kwargs):
        pass

    def create_table(self, table_name, *args, **kwargs):
        pass

    def drop_table(self, table_name, *args, **kwargs):
        pass

    def insert(self, table_name, items, *args, **kwargs):
        pass

    def delete(self, table_name, *args, **kwargs):
        pass

    def update(self, table_name, *args, **kwargs):
        pass

    def update_insert(self, table_name, *args, **kwargs):
        pass

    def find(self, table_name, *args, **kwargs):
        pass

    def find_all(self, table_name, cols=None):
        pass

    def find_by_col(self, table_name, col_name, val):
        pass

    def find_by_id(self, table_name, id):
        pass

    def find_max(self, table_name, query, col_name):
        pass

    def find_min(self, table_name, query, col_name):
        pass

    def to_pandas(self, val, *args, **kwargs):
        pass

    def get_id(self, db_obj):
        pass


class MongoDBClient(DBClient, metaclass=Singleton):
    def __init__(self, db_name, *args, **kwargs):
        self.client = MongoClient(*args, **kwargs)
        self.db = self.client[db_name]

    def create_table(self, table_name):
        return self.db[table_name]

    def drop_table(self, table_name, *args, **kwargs):
        return self.db[table_name].drop(*args, **kwargs)

    def insert(self, table_name, items, time_logs=False, *args, **kwargs):
        table = self.db[table_name]

        time_logs_obj = {}

        if time_logs:
            time_logs_obj = {
                "created": datetime.utcfromtimestamp(time())
            }

        if type(items) == list:
            return table.insert_many(
                [{**item, **time_logs_obj} for item in items], *args, **kwargs)
        elif type(items) == dict or type(items) == OrderedDict:
            return table.insert_one(
                {**items, **time_logs_obj}, *args, **kwargs)
        else:
            raise Warning("Failed to add items")
            return None

    def delete(self, table_name, query=None, items=None, *args, **kwargs):
        if items is None and query is None:
            raise Warning("No items or query is specified. No deletions")
            return None

        if query:
            table = self.db[table_name]
            return table.delete_many(query, *args, **kwargs)

        if items:
            if type(items) == dict:
                return table.delete_one({'_id', items['_id']}, *args, **kwargs)
            elif type(items) == list:
                expresssions = [{'_id': item['_id']} for item in items]
                return table.delete_many({
                    '$or': expresssions
                })
            else:
                raise Warning("No deletions")
                return None

    def update(self, table_name, query, new_val, time_logs=False,
               *args, **kwargs):
        table = self.db[table_name]

        time_logs_obj = {}

        if time_logs:
            time_logs_obj = {
                '$currentDate': {
                    'lastModified': True
                }
            }
        return table.update_many(
            query, {'$set': new_val, **time_logs_obj}, *args, **kwargs)

    def update_insert(self, table_name, query, new_val, *args, **kwargs):
        table = self.db[table_name]

        return table.update_one(query, {'$set': new_val}, upsert=True)

    def update_by_id(self, table_name, id, new_val, time_logs=False):
        table = self.db[table_name]

        time_logs_obj = {}

        if time_logs:
            time_logs_obj = {
                '$currentDate': {
                    'lastModified': True
                }
            }

        return table.update_one(
            {'_id': id}, {'$set': new_val, **time_logs_obj})

    def find(self, table_name, query=None, projection=None, *args, **kwargs):
        table = self.db[table_name]

        return table.find(query or {}, projection or {},
                          *args, **kwargs)

    def find_all(self, table_name, cols=None, id=True):
        table = self.db[table_name]
        if cols is None:
            if id:
                return table.find()
            else:
                return table.find({}, {'_id': 0})
        else:
            cols_dict = {col: 1 for col in cols}
            return table.find({}, {'_id': 1 if id else 0, **cols_dict})

    def find_by_col(self, table_name, col_name, val):
        table = self.db[table_name]

        return table.find({col_name: val})

    def find_by_id(self, table_name, id):
        table = self.db[table_name]

        return table.find_one({'_id': id})

    def find_max(self, table_name, query, col_name):
        table = self.db[table_name]

        return table.find(query).sort({col_name: -1}).limit(1)

    def find_min(self, table_name, query, col_name):
        table = self.db[table_name]

        return table.find(query).sort({col_name: 1}).limit(1)

    def to_pandas(self, val, id=True):
        df = pd.DataFrame(list(val))

        # Delete _id
        if not id:
            del df['_id']

        return df

    def get_id(self, db_obj):
        return db_obj['_id']
