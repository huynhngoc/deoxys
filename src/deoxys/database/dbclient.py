# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"


from ..utils import Singleton
from pymongo import MongoClient
from bson.objectid import ObjectId
from collections import OrderedDict
from datetime import datetime
from time import time
import json
import pandas as pd


class DBClient(metaclass=Singleton):  # pragma: no cover
    def __init__(self, *args, **kwargs):
        pass

    def create_table(self, table_name, *args, **kwargs):
        raise NotImplementedError

    def drop_table(self, table_name, *args, **kwargs):
        raise NotImplementedError

    def insert(self, table_name, items, *args, **kwargs):
        raise NotImplementedError

    def delete(self, table_name, *args, **kwargs):
        raise NotImplementedError

    def update(self, table_name, *args, **kwargs):
        raise NotImplementedError

    def update_insert(self, table_name, *args, **kwargs):
        raise NotImplementedError

    def find(self, table_name, *args, **kwargs):
        raise NotImplementedError

    def find_all(self, table_name, cols=None):
        raise NotImplementedError

    def find_by_col(self, table_name, col_name, val):
        raise NotImplementedError

    def find_by_id(self, table_name, id):
        raise NotImplementedError

    def find_max(self, table_name, query, col_name):
        raise NotImplementedError

    def find_min(self, table_name, query, col_name):
        raise NotImplementedError

    def to_pandas(self, val, *args, **kwargs):
        raise NotImplementedError

    def get_id(self, db_obj):
        raise NotImplementedError

    def df_to_json(self, df):
        cols = df.columns

        data = []
        for item in df.values:
            item_dict = {}
            for i in range(len(cols)):
                try:
                    json.dumps(item[i])
                    item_dict[cols[i]] = item[i]
                except Exception:
                    item_dict[cols[i]] = str(item[i])
            data.append(item_dict)
        return data


class MongoDBClient(DBClient, metaclass=Singleton):  # pragma: no cover
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

        if type(val) == list:
            expresssions = [{col_name: v} for v in val]
            return table.find({
                '$or': expresssions
            })

        return table.find({col_name: val})

    def find_by_id(self, table_name, id, custom_id=False):
        table = self.db[table_name]

        if type(id) == list:
            expresssions = [{'_id': self._serialize_id(id_) for id_ in id}]
            return table.find({
                '$or': expresssions
            })
        else:
            return table.find_one({'_id': self._serialize_id(id)})

    def _serialize_id(self, id, custom_id=False):
        if custom_id:
            return id
        if type(id) == str:
            return ObjectId(id)
        return id

    def to_fk(self, val):
        def str_to_id(v):
            if type(v) == str:
                return ObjectId(v)
            return v
        if type(val) == list:
            return [str_to_id(v) for v in val]
        if type(val) == str:
            return str_to_id(val)
        else:
            return val

    def find_max(self, table_name, query, col_name):
        table = self.db[table_name]

        res = table.find(query).sort(col_name, -1).limit(1)
        if res.count() > 0:
            return res[0]
        else:
            return None

    def find_min(self, table_name, query, col_name):
        table = self.db[table_name]

        res = table.find(query).sort(col_name, 1).limit(1)[0]

        if res.count() > 0:
            return res[0]
        else:
            return None

    def to_pandas(self, val, id=True):
        df = pd.DataFrame(list(val))

        # Delete _id
        if not id:
            del df['_id']

        return df

    def get_id(self, db_obj):
        return db_obj['_id']
