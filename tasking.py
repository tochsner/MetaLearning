import thread

def run(i):
    print(i)

for i in range(100):
    thread.start_new_thread(run)

thread.join()