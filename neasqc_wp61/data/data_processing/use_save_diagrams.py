import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../../models/quantum/pre_alpha_2/")
from pre_alpha_2 import *
import argparse

def main():
    """
    Saves the diagrams of reduced_amazonreview_pre_alpha_train.tsv and 
    reduced_amazonreview_pre_alpha_test.tsv as pickle objects
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--train", help = "Path of train dataset")
    parser.add_argument("-val", "--validation", help = "Path of validation dataset")
    parser.add_argument("-te", "--test", help = "Path of test dataset")
    parser.add_argument("-o", "--output", help = "Output path") 
    args = parser.parse_args()
    train_dataset_name = os.path.basename(args.train)
    train_dataset_name = os.path.splitext(train_dataset_name)[0]
    validation_dataset_name = os.path.basename(args.validation)
    validation_dataset_name = os.path.splitext(validation_dataset_name)[0]
    test_dataset_name = os.path.basename(args.test)
    test_dataset_name = os.path.splitext(test_dataset_name)[0]

    sentences_train = PreAlpha2.load_dataset(args.train)[0]
    sentences_validation = PreAlpha2.load_dataset(args.validation)[0]
    sentences_test = PreAlpha2.load_dataset(args.test)[0]
    diagrams_train = PreAlpha2.create_diagrams(sentences_train)
    diagrams_validation = PreAlpha2.create_diagrams(sentences_validation)
    diagrams_test = PreAlpha2.create_diagrams(sentences_test)


    PreAlpha2.save_diagrams(
        diagrams_train, args.output, 'diagrams_' + train_dataset_name 
    )
    PreAlpha2.save_diagrams(
        diagrams_validation, args.output, 'diagrams_' + validation_dataset_name 
    )
    PreAlpha2.save_diagrams(
        diagrams_test, args.output, 'diagrams_' + test_dataset_name 
    )


if __name__ == "__main__":
    main()