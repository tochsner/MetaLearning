import pickle
from time import sleep
from timeit import default_timer as timer
from threading import Thread

class HistoryItem:

    def __init__(self, rule, accuracy_history, produced_invalid_values):
        self.rule = rule
        self.accuracy_history = accuracy_history
        self.produced_invalid_values = produced_invalid_values

class LoggingThread(Thread):
        def __init__(self, queue, path, total_num_items):
            Thread.__init__(self)
            self.queue = queue
            self.path = path
            self.total_num_items = total_num_items

            self.ended = False
        
        def run(self):      
            with open(self.path, 'wb') as file:
                pass

            max_accuracy = 0
        
            start_time = timer()
            seconds_per_element = 0
            minutes_left = 0

            elements_processed = 0

            while True:                                
                while self.queue.empty():
                    sleep(60)

                history = []
                
                while not self.queue.empty():
                    history_item = self.queue.get()
                    history.append(history_item)   

                    elements_processed += 1

                    accuracy = max(history_item.accuracy_history)
                    max_accuracy = max(max_accuracy, accuracy)                 

                    print('{:0>5d}'.format(elements_processed),
                            '/', '{:0>5d}'.format(self.total_num_items),
                            'Accuracy:',
                            '{:.2f}'.format(accuracy), 
                            '(Max:', '{:.2f}'.format(max_accuracy), ')',
                            '; Time per Rule:',
                            '{:.2f}'.format(seconds_per_element),
                            '; Time Left:',
                            '{:.1f}'.format(minutes_left), 'min',
                            '; Used Rule:',
                            str(history_item.rule))
                
                time = timer()
                seconds_per_element = (time - start_time) / elements_processed
                minutes_left = (self.total_num_items - elements_processed) * seconds_per_element / 60

                try:
                    with open(self.path, 'ab') as file:
                            pickle.dump(history, file)
                except:
                    pass

                if self.ended:
                    return

class HistoryManager:
    
    def __init__(self, queue, path, total_num_items):
        self.thread = LoggingThread(queue, path, total_num_items)
        self.thread.start()

    def end(self):
        self.thread.ended = True
        self.thread.join()
