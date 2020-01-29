from deoxys.database import MongoDBClient
from deoxys.experiment import ExperimentDB
from deoxys.web.app import VisApp


if __name__ == '__main__':
    dbclient = MongoDBClient('deoxys', 'localhost', 27017)
    app = VisApp(dbclient, '../../../web')
    app.run()
