import traceback
import sys


def show_exception_file(func):
    def _wrapper(*args, **kwargs):
        try:
            _result = func(*args, **kwargs)
            return _result
        except Exception as e:
            except_info = str(e)
            batch_pack = kwargs['batch_pack']
            file_name = f'({batch_pack["image_name"]}, {batch_pack["label_name"]})'
            tb = sys.exc_info()[2]
            raise type(e)(f'{except_info}\nWhen training file(image, mask): {file_name}').with_traceback(tb)
    return _wrapper
