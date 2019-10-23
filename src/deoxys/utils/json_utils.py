import json


def load_data_from_json(json_content):
    content = json.loads(json_content)

    if 'type' in content and 'structure' in content:
        return content['type'], content['structure']
    else:
        raise ValueError()
