#Read input file
#   parameters:#       input file - .i input file
# learning_rate_function  = 'step'  # Type of learning rate function (options are: step, exponential, or linear)


def is_number(s):
    """Determine if the string should be a float

    Args:
        s (str): Input String

    Returns:
        bool: Boolean determining if the string should be a float
    """
    try:
        float(s)
        return True
    except ValueError:
        return False
    

def read_input_file(input_file):
    inputs = {}
    with open(input_file) as f:
        for line in f:
            if '=' in line:  # Only read in lines that have an equals sign
                value = line.split("=")[1].strip()  # Get value
                value = value.split("#")[0].strip()  # Get rid of any comment
                inputs[line.split("=")[0].strip()] = value
            else: pass
        for key,val in inputs.items():  # Turn any numbers into a float from a string
            if key in ['decay', 'learning_rate']:
                inputs[key] = float(val)
            if key in ['nodes', 'conv_filters', 'kernel_size', 'epochs', 't_steps', 'batch_size', 'epochs_drop']:  # Make these integers
                inputs[key] = int(val)
        
    return inputs