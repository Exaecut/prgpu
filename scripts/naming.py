import re

def enforce_pascal(name: str):
    if not re.match(r"^[A-Z][A-Za-z0-9]*$", name):
        raise ValueError("Name must be PascalCase (e.g. ExposureBlur)")


def to_snake(name: str) -> str:
    result = ""
    for i, c in enumerate(name):
        if c.isupper():
            if i > 0:
                prev = name[i - 1]
                if prev.islower() or prev.isdigit():
                    result += "_"
                elif prev.isupper() and i < len(name) - 1 and name[i + 1].islower():
                    result += "_"
            result += c.lower()
        else:
            result += c
    return result


def to_upper_flat(name: str) -> str:
    return name.upper()
