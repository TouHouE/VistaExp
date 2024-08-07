import json
import os


def load_labels(path, encoding='utf-8') -> dict[int, str]:
    with open(path, 'r', encoding=encoding) as jin:
        mapper = json.load(jin)
    mapper = {int(key): value for key, value in mapper.items()}
    return mapper


def save_json(path, content, encoding='utf-8'):
    with open(path, 'w+', encoding=encoding) as jout:
        json.dump(content, jout)


def save_continue_json(path, new_content, encoding='utf-8'):
    if not os.path.exists(path):
        save_json(path, new_content, encoding)
        return
    old_content = load_labels(path)
    if isinstance(old_content, list):
        old_content.append(new_content)
    elif isinstance(old_content, dict):
        old_content.update(new_content)
    save_json(path, old_content, encoding)



