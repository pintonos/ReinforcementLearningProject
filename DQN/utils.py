
def log(logfile, message, console=True):
    """
    A simple logging utility function.
    """
    with open(logfile, 'a+') as f:
        f.write(message)
        f.write('\n')
        if console:
            print(message)
