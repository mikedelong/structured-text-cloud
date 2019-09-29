def get_setting(arg_name, arg_setting):
    result = arg_setting[arg_name] if arg_name in arg_setting.keys() else None
    return result
