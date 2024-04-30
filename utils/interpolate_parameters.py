import numpy as np


def interpolate_array(data, fit_count):
    """Interpola linealmente los datos a un nuevo tamaño de array."""
    linear_interpolate = lambda before, after, at_point: before + (after - before) * at_point

    spring_factor = (len(data) - 1) / (fit_count - 1)
    new_data = np.zeros(fit_count)
    new_data[0] = data[0]  # asignación inicial

    for i in range(1, fit_count - 1):
        tmp = i * spring_factor
        before = int(np.floor(tmp))
        after = int(np.ceil(tmp))
        at_point = tmp - before
        new_data[i] = linear_interpolate(data[before], data[after], at_point)

    new_data[-1] = data[-1]  # asignación final
    return new_data.tolist()


def interpolate_params(param_list, sampling_rate, audio_length):
    """Interpola una lista de parámetros para coincidir con el número de muestras necesario."""
    if audio_length == 0:
        assert False, "Audio length cannot be zero."

    if not param_list:
        assert False, "Parameter list cannot be empty."

    number_of_samples = int(sampling_rate * audio_length / 512 + 1)
    interpolated_params = []

    for params in param_list:
        if len(params) == 1:
            interpolated_params.append(np.full(number_of_samples, params[0]))
        else:
            interpolated_params.append(interpolate_array(params, number_of_samples))

    return interpolated_params
