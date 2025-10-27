import csv
import numpy as np
import os
import yaml
from rcwa.utils import nk_location

def nk_to_complex(data, column_labels=None):
    """
    Convert tabulated n/k measurements into complex refractive index arrays.

    Parameters
    ----------
    data:
        Numeric array whose first column contains wavelengths and the remaining
        columns hold refractive index (and optionally extinction coefficient).
    column_labels:
        Optional list of column headers. If provided and the first header
        mentions nanometres we convert wavelengths to micrometres to maintain
        compatibility with the historical pandas-based loader.
    """

    if not isinstance(data, np.ndarray):
        raise NotImplementedError

    wavelengths = data[:, 0]
    if column_labels:
        header = column_labels[0].lower()
        if "(nm" in header or " nanometer" in header:
            wavelengths = wavelengths / 1000.0

    if data.shape[1] == 3:
        nk_complex = data[:, 1] + 1j * data[:, 2]
    elif data.shape[1] == 2:
        nk_complex = data[:, 1]
    else:
        raise ValueError("Expected two or three columns of nk data")

    return wavelengths, nk_complex

class CSVLoader:
    def __init__(self, filename):
        self.filename = filename

    def load(self):
        header = None
        numeric_rows = []

        def is_numeric(row):
            try:
                [float(cell) for cell in row]
            except ValueError:
                return False
            return True

        with open(self.filename, newline='', encoding='utf-8-sig') as handle:
            reader = csv.reader(handle)
            for raw_row in reader:
                row = [cell.strip() for cell in raw_row]
                if not row or all(cell == '' for cell in row):
                    continue
                if row[0].startswith('#'):
                    continue
                if header is None and not is_numeric(row):
                    header = row
                    continue
                if not is_numeric(row):
                    continue
                numeric_rows.append([float(cell) for cell in row])

        if not numeric_rows:
            raise ValueError(f'No numeric data found in {self.filename}')

        raw_data = np.array(numeric_rows, dtype=np.float64)
        wavelengths, n_dispersive = nk_to_complex(raw_data, header)
        er_dispersive = np.sqrt(n_dispersive)
        ur_dispersive = np.ones(er_dispersive.shape)

        return {'wavelength': wavelengths, 'n': n_dispersive, 'er': er_dispersive, 'ur': ur_dispersive}

class RIDatabaseLoader:

    def __init__(self):
        """
        Loader for RefractiveIndex.info databases
        """
        self.extract_material_database()

    def extract_material_database(self):
        database_filename = os.path.join(nk_location, 'library.yml')
        self.materials = {}
        with open(database_filename, encoding='utf-8') as database_file:
            database_list = yaml.load(database_file, Loader=yaml.FullLoader)

        main_content = database_list[0]['content']
        for i in range(len(main_content)):
            book_or_divider = database_list[0]['content'][i]

            if 'BOOK' in book_or_divider.keys():
                material = main_content[i]['BOOK']
                material_content = main_content[i]['content']

                for j in range(len(material_content)):
                    if 'PAGE' in material_content[j].keys():
                        file_location = material_content[j]['data']
                        self.materials[material] = file_location
                        break

    def load(self, filename):
        with open(filename, encoding='utf-8') as fn:
            material_file = yaml.load(fn, Loader=yaml.FullLoader)
            material_data = material_file['DATA'][0]
            if material_data['type'] == 'tabulated nk':
                data = self.load_nk_table_data(material_data)
            else:
                data = self.load_nk_formula_data(material_data)

        return data

    def load_nk_formula_data(self, data_dict):
        if data_dict['type'] == 'formula 1':
            return self.load_nk_formula_1_data(data_dict)
        elif data_dict['type'] == 'formula 2':
            return self.load_nk_formula_2_data(data_dict)
        else:
            raise ValueError(f'Formula type {data_dict["type"]} not supported. Please submit a bug report with this message and the specific material you are trying to use')

    def load_nk_formula_1_data(self, data_dict):
        coeffs = data_dict['coefficients'].split()
        coeffs = [float(x) for x in coeffs]
        A = coeffs[0]
        num_terms = int((len(coeffs) - 1) / 2)
        B_coeffs = [coeffs[2*i+1] for i in range(num_terms)]
        C_coeffs = [coeffs[2*i+2] for i in range(num_terms)]

        def dispersion_formula_er(wavelength):
            L = wavelength
            b_terms = [b * L**2 / (L**2 - c**2) for b, c in zip(B_coeffs, C_coeffs)]
            full_term = 1 + A + np.sum(b_terms)
            return full_term

        def dispersion_formula_n(wavelength):
            return np.sqrt(dispersion_formula_er(wavelength))
        def dispersion_formula_ur(wavelength):
            return 1

        return {'er': dispersion_formula_er, 'ur': dispersion_formula_ur, 'n': dispersion_formula_n,
                'dispersion_type': 'formula'}

    def load_nk_formula_2_data(self, data_dict):
        coeffs = data_dict['coefficients'].split()
        coeffs = [float(x) for x in coeffs]
        A, B1, C1, B2, C2 = coeffs
        def dispersion_formula_er(wavelength):
            b1_term = B1 * wavelength **2 / (wavelength**2 - C1)
            b2_term = B2 * wavelength**2 / (wavelength**2 - C2)
            full_term = 1 + A + b1_term + b2_term
            return full_term
        def dispersion_formula_n(wavelength):
            return np.sqrt(dispersion_formula_er(wavelength))
        def dispersion_formula_ur(wavelength):
            return 1

        return {'er': dispersion_formula_er, 'ur': dispersion_formula_ur, 'n': dispersion_formula_n,
                'dispersion_type': 'formula'}


    def load_nk_table_data(self, data_dict):
        material_data = data_dict['data']
        nk_data_string = list(filter(None, material_data.split('\n')))
        split_data = [elem.split() for elem in nk_data_string]
        numerical_data = np.array(split_data, dtype=np.float64)

        wavelengths, n_dispersive = nk_to_complex(numerical_data)

        return {'er': np.square(n_dispersive), 'ur': np.ones(n_dispersive.shape), 'n': n_dispersive,
                'dispersion_type': 'tabulated', 'wavelength': wavelengths}
