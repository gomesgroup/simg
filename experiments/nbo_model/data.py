import os


def split(data_list, directory):
    def load_indexes(path):
        with open(path, 'rt') as f:
            indexes = list(map(int, f.read().strip().split('\n')))

        return indexes

    train = load_indexes(os.path.join(directory, 'train.txt'))
    val = load_indexes(os.path.join(directory, 'val.txt'))
    test = load_indexes(os.path.join(directory, 'test.txt'))

    train = list(filter(lambda x: x < len(data_list), train))
    val = list(filter(lambda x: x < len(data_list), val))
    test = list(filter(lambda x: x < len(data_list), test))

    return [data_list[i] for i in train], [data_list[i] for i in val], [data_list[i] for i in test]
