from wiktionaryparser import WiktionaryParser

if __name__ == '__main__':
    parser = WiktionaryParser()
    result = parser.fetch('should')
    for definition in result[0]['definitions']:
        print(definition['partOfSpeech'])
