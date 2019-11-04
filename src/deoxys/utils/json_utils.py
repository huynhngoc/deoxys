import json


def load_json_config(*args):
    data = []
    for arg in args:
        if arg is None:
            data.append(arg)
        elif type(arg) == dict:
            data.append(arg)
        elif type(arg) == str:
            try:
                data.append(json.loads(arg))
            except json.JSONDecodeError:
                data.append(None)
                raise Warning('Decode JSON failed. Return None instead.')
        else:
            data.append(None)
            raise Warning('Invalid datatype. Return None instead.')

    if len(data) == 1:
        return data[0]
    return data
