
def parse(layers_param):
    try:
        value = int(layers_param)
        return value, None
    except ValueError:
        instruction = layers_param
        return None, instruction


def extract_dims(layers_params, current_index):
    current_value = int(layers_params[current_index])
    prev_value = None
    not_found = True

    i = 0
    while not_found:
        try:
            i = i + 1
            prev_value = int(layers_params[current_index - i])
            not_found = False
        except ValueError:
            continue  # Try next value/string int the list

    return prev_value, current_value