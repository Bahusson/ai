# Wzięte z: https://stackoverflow.com/questions/39358092/range-as-dictionary-key-in-python
class RangeDict(dict):
    """Pozwala na użycie zakresu jako klucza w słowniku"""
    def __getitem__(self, item):
        if type(item) != range: # or xrange in Python 2
            for key in self:
                if item in key:
                    return self[key]
        else:
            return super().__getitem__(item)

def HLayer(self,hlay_num,hlay_size,hlay_diff):
    """Pozwala na oszczędzenie kodu przy tworzeniu kolejnych warstw ukrytych"""
    class varname(object): # Tworzy klasę-obiekt (tj. nową zmienną) jak u tego Hindusa w przykładzie input_data (spyder)
                pass       # czyli to czego ogólnie rzecz biorąc mi potrzeba, tylko jeszcze nie wiem jak to ogarnę.
                           # pomoc: https://stackoverflow.com/questions/100003/what-are-metaclasses-in-python/6581949?r=SearchResults#6581949
    # Wymaga być może dopracowania do metaklasy, bo potrzebuję tu tworzyć X nowych zmiennych a średnio mam na to teraz czas.
    def __init__(self): # Tu pójdzie dodatkowy parametr "hlay_num"
        for name in range(hlay_num):
            classname = varname()

    def generate(hlay_num,hlay_size,hlay_diff):
        for layer in range(hlay_num):
            if layer in hlay_diff:
                varname[layer] = tf.get_variable("weights_2",[hlay_diff[layer], hlay_diff[layer])
                biases_2 = tf.get_variable("biases_2",[hlay_diff[layer])
                outputs_1m = tf.nn.relu(tf.matmul(outputs_1m, weights_2) + biases_2)
            else:
                weights_2 = tf.get_variable("weights_2",[hlay_size, hlay_size])
                biases_2 = tf.get_variable("biases_2",[hlay_size])
            outputs_2 = tf.nn.relu(tf.matmul(outputs_1m, weights_2) + biases_2)
            #Funkcja mieli tak