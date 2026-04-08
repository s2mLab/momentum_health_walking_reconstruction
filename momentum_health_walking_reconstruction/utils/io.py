import os


class Io:
    @staticmethod
    def parse_vector_env_variable(variable_name: str, minimum_length: int = None, maximum_length: int = None):
        variable = os.getenv(variable_name)
        if variable is None:
            raise ValueError(f"Environment variable '{variable_name}' is not set.")

        items = [item.strip() for item in variable.split(",") if item.strip()]
        if minimum_length is not None and len(items) < minimum_length:
            raise ValueError(f"Environment variable '{variable_name}' must contain at least {minimum_length} items.")
        if maximum_length is not None and len(items) > maximum_length:
            raise ValueError(f"Environment variable '{variable_name}' must contain at most {maximum_length} items.")
        return items

    @staticmethod
    def parse_multivectors_env_variable(
        variable_name: str,
        inner_minimum_length: int = None,
        inner_maximum_length: int = None,
        outer_minimum_length: int = None,
        outer_maximum_length: int = None,
    ):
        variable = os.getenv(variable_name)
        if variable is None:
            raise ValueError(f"Environment variable '{variable_name}' is not set.")

        multivectors = []
        for vector in variable.split(";"):
            items = [item.strip() for item in vector.strip("[]").split(",") if item.strip()]
            if inner_minimum_length is not None and len(items) < inner_minimum_length:
                raise ValueError(
                    f"Each vector in environment variable '{variable_name}' must contain at least {inner_minimum_length} items."
                )
            if inner_maximum_length is not None and len(items) > inner_maximum_length:
                raise ValueError(
                    f"Each vector in environment variable '{variable_name}' must contain at most {inner_maximum_length} items."
                )
            if items:
                multivectors.append(items)
        if outer_minimum_length is not None and len(multivectors) < outer_minimum_length:
            raise ValueError(
                f"Environment variable '{variable_name}' must contain at least {outer_minimum_length} vectors."
            )
        if outer_maximum_length is not None and len(multivectors) > outer_maximum_length:
            raise ValueError(
                f"Environment variable '{variable_name}' must contain at most {outer_maximum_length} vectors."
            )
        return multivectors
