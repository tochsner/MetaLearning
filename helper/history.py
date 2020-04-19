import pickle
from time import sleep
from timeit import default_timer as get_current_time
from threading import Thread


"""
A HistoryItem describes a evaluated update rule and its results in the exhaustive search.

The accuracy history is a list containing the achieved accuracies in each epoch.
"""
class HistoryItem:

    def __init__(self, rule, accuracy_history, produced_invalid_values=False):
        self.rule = rule
        self.accuracy_history = accuracy_history
        self.produced_invalid_values = produced_invalid_values
        self.test_accuracy = 0

    def get_best_accuracy(self):
        return max(self.accuracy_history)


"""
The LoggingThread class periodically stores the results of an exhaustive search on disk.

It is designed to run in a separate thread and incrementally appends the evaluated 
rules to the logfile. After every store, it goes to sleep for 1 minute.

Also works on existing log files as it only appends the new results.
"""
class LoggingThread(Thread):

        def __init__(self, queue, path, total_num_items):
            Thread.__init__(self)
            self.queue = queue
            self.path = path
            self.total_num_items = total_num_items

            self.best_accuracy = 0

            self.ended = False


        """
        Runs until self.ended is set to True.
        Wakes up every minute and logs the results. 
        """
        def run(self):      
            start_time = get_current_time()

            elements_processed = 0

            while True:
                while self.queue.empty():
                    if self.ended and self.queue.empty():
                        print("Logging process ended.")
                        return
                    sleep(60)

                recent_history_items = []
                
                while not self.queue.empty():
                    history_item = self.queue.get()
                    recent_history_items.append(history_item)

                current_time = get_current_time()
                elements_processed += len(recent_history_items)
                seconds_per_element = (current_time - start_time) / elements_processed
                minutes_left = (self.total_num_items - elements_processed) * seconds_per_element / 60

                best_recent_rule = max(recent_history_items, key=HistoryItem.get_best_accuracy)
                best_recent_accuracy = best_recent_rule.get_best_accuracy()
                self.best_accuracy = max(self.best_accuracy, best_recent_accuracy)

                for history_item in recent_history_items:
                    print('{:0>5d}'.format(elements_processed),
                            '/', '{:0>5d}'.format(self.total_num_items),
                            'Accuracy:',
                            '{:.2f}'.format(history_item.get_best_accuracy()),
                            '(Max:',
                            '{:.2f}'.format(self.best_accuracy), ')',
                            '; Time per Rule:',
                            '{:.2f}'.format(seconds_per_element),
                            '; Time Left:',
                            '{:.1f}'.format(minutes_left), 'min',
                            '; Used Rule:',
                            str(history_item.rule))

                try:
                    with open(self.path, 'ab') as file:
                        pickle.dump(recent_history_items, file)
                except Exception as e:
                    print(e)


"""
This class starts and ends a LoggingThread.
"""
class HistoryManager:
    
    def __init__(self, queue, path, total_num_items):
        self.thread = LoggingThread(queue, path, total_num_items)
        self.thread.start()

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.thread.ended = True
        self.thread.join()
