
def log(logfile, message, console=True):
    with open(logfile, 'a+') as f:
        f.write(message)
        f.write('\n')
        if console:
            print(message)
