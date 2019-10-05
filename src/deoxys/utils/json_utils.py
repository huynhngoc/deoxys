def read_file(filename):
    """
    Return the content of a file

    :param filename: file name
    :type filename: str
    :return: content of the file
    :rtype: str
    """
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()


def write_file(content, filename):
    """
    Write to a file

    :param content: content of the file
    :type content: str
    :param filename: file name
    :type filename: str
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
