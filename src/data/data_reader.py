import scipy.io
import os


class DataReader:
    def __init__(self, data_folder_path) -> None:
        self.data_folder_path = data_folder_path
    
    def read_data(self, D: int = 1, K0: int = 10):
        data = None
        filename = f"Data_D_{D}_K0_{K0}.mat"
        path = os.path.join(self.data_folder_path, filename)
        if os.path.exists(path):
            data = scipy.io.loadmat(path)

            Cq = data["Cfc"]
            Cq_plus = data["Cfcp"]
            Z_set = data["Z_set"][0]
            T_span = data["T_span"][0]
            Current = data["G"][0]
            Potential = data["pot"][0]
            K0 = data["K0"][0, 0]
            D = data["D"][0, 0]

            data = {
                "Cq": Cq,
                "Cq_plus": Cq_plus,
                "Z_set": Z_set,
                ""
            }

        return data
