def get_model(model_name, args, algorithm=None):
    algorithm = str(algorithm or args.get("algorithm", "rsiat")).lower()
    name = model_name.lower()
    if algorithm == "rsiat" and name == "adapter":
        from models.RSIAT_adapter import Learner
        return Learner(args)
    if algorithm != "rsiat":
        raise ValueError("Unsupported algorithm {!r}.".format(algorithm))
    raise ValueError("Unsupported model_name {!r} for {}.".format(model_name, algorithm))
