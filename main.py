import json

if __name__ == '__main__':
    with open('./settings.json') as settings_fp:
        settings = json.load(settings_fp)
        print(settings)
