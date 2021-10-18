import json
from . import read_file


def load_json_config(*args):
    data = []
    for arg in args:
        if arg is None:
            data.append(arg)
        elif type(arg) == dict or type(arg) == list:
            data.append(arg)
        elif type(arg) == str:
            try:
                if arg.startswith('{') or arg.startswith('['):
                    data.append(json.loads(arg))
                else:
                    data.append(json.loads(read_file(arg)))
            except json.JSONDecodeError:
                data.append(None)
                raise Warning('Decode JSON failed. Return None instead.')
            except IOError:
                data.append(None)
                raise Warning('File not found. Return None instead.')
            except Exception as ex:
                data.append(None)
                raise Warning('An error occur. {}'.format(ex))
        else:
            data.append(None)
            raise Warning('Invalid datatype. Return None instead.')

    if len(data) == 1:
        return data[0]
    return data
