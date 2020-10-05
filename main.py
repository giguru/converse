# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import haystack

USE_GPU = True

from haystack.reader.farm import FARMReader

reader = FARMReader(model_name_or_path="distilbert-base-uncased-distilled-squad", use_gpu=USE_GPU)
# train_data = "data/squad20"
# train_data = "PATH/TO_YOUR/TRAIN_DATA"
# reader.train(data_dir=train_data, train_filename="dev-v2.0.json", use_gpu=True, n_epochs=1, save_dir="my_model")

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
