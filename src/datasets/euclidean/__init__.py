class DatasetRegistry:
    """A registry to hold dataset classes."""
    _datasets = {}

    @classmethod
    def register(cls, name=None):
        """
        A decorator for registering dataset classes.

        Args:
            name (str, optional): The name to register the class under. If not provided, the class name is used.

        Returns:
            Callable: The decorated class.
        """

        def decorator(dataset_cls):
            reg_name = name or dataset_cls.__name__
            if reg_name in cls._datasets:
                raise ValueError(f"Dataset '{reg_name}' is already registered.")
            cls._datasets[reg_name] = dataset_cls
            return dataset_cls
        return decorator

    @classmethod
    def get(cls, name):
        """
        Retrieve a dataset class by name.

        Args:
            name (str): The name of the dataset class to retrieve.

        Returns:
            type: The dataset class.
        """
        if name not in cls._datasets:
            raise KeyError(f"Dataset '{name}' is not registered.")
        return cls._datasets[name]

    @classmethod
    def list_datasets(cls):
        """
        List all registered dataset names.

        Returns:
            list: A list of registered dataset names.
        """
        return list(cls._datasets.keys())