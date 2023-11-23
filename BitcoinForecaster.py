from Model import Model

if __name__ == '__main__':
    m = Model()
    print(format(m.forecast().values[0], ','))