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
