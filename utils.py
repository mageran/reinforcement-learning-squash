import sys
from termcolor import colored

text_colors = [
    "grey",
    "red",
    "green",
    "yellow",
    "blue",
    "magenta",
    "cyan",
    "white",
]

highlights = [
    "on_grey",
    "on_red",
    "on_green",
    "on_yellow",
    "on_blue",
    "on_magenta",
    "on_cyan",
    "on_white",
]

attributes = [
    "bold",
    "dark",
    "underline",
    "blink",
    "reverse",
    "concealed",
]

def create_colored_print_functions():
    """Dynamically creates print functions for all combinations."""
    globals_dict = globals()  # Access the global scope

    for color in text_colors:
        for highlight in [None] + highlights:  # Include None for no highlight
            for attr_combination in [()] + [tuple([attr]) for attr in attributes] + [tuple([attr1, attr2]) for attr1 in attributes for attr2 in attributes if attr1 != attr2] + [tuple([attr1,attr2,attr3]) for attr1 in attributes for attr2 in attributes for attr3 in attributes if attr1!=attr2 and attr1!=attr3 and attr2!=attr3]:
                function_name = f"print_{color}"
                if highlight:
                    function_name += f"_{highlight[3:]}" #remove the "on_"
                if attr_combination:
                    function_name += "_" + "_".join(attr_combination)

                def create_print_function(color, highlight, attr_combination):
                    def colored_print(text, **kwargs):
                        print(colored(text, color, highlight, attrs=list(attr_combination)), **kwargs)
                    return colored_print

                globals_dict[function_name] = create_print_function(color, highlight, attr_combination)

create_colored_print_functions()
