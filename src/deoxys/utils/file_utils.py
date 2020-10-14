
import os


def read_file(filename):
    """
    Return the content of a file

    Parameters
    ----------
    filename : str
        file name

    Returns
    -------
    str
        content of the file
    """
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()


def write_file(content, filename):
    """
    Write to a file

    Parameters
    ----------
    filename : str
        file name

    content: str
        content of the file
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)


def write_byte(content, filename):
    """
    Write bytes to a file

    Write to a file

    Parameters
    ----------
    filename : byte
        file name

    content: str
        content of the file
    """
    with open(filename, "wb") as f:
        f.write(content)


def file_finder(filename, callback=None, **kwargs):   # pragma: no cover
    # check for file existence
    if os.path.isfile(filename):
        return filename
    # use callback to perform customized actions
    elif callable(callback):
        return callback(**kwargs)
    # default behavior
    else:
        # Choose between ignoring the file or enter a new filename
        msg = '"{}" not fould! Ignore?[y/n]: '.format(filename)
        if input(msg).upper() != 'Y':
            count = 0
            while not os.path.isfile(filename) and count < 3:
                msg = '"{}" not fould! Please enter the new filename: '.format(
                    filename)
                filename = input(msg)
                count += 1
            if count < 3:
                return filename
        print('File ignored!')

        return None
