from modulab.operators.base import Operator


class Passthrough(Operator):
    name = "passthrough"

    def apply(self, enc):
        return enc

    def compatible(self, repr_name):
        return True
