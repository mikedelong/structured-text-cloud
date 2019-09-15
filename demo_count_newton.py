from scipy.optimize import newton
from sklearn.feature_extraction.text import CountVectorizer


def f(arg):
    return -2.0 + arg / 2.0


def get_count(arg):
    documents = ['the rain in Spain falls mainly in the plane',
                 'The Tahiti rail, Tahitian red-billed rail, or Pacific red-billed rail (Gallirallus pacificus) is an'
                 ' extinct species of rail that lived on Tahiti. It was first recorded during James Cook\'s second'
                 ' voyage around the world (1772–1775), on which it was illustrated by Georg Forster and described by'
                 ' Johann Reinhold Forster.',
                 ' No specimens have been preserved. As well as the documentation by the'
                 ' Forsters, there have been claims that the bird also existed on the nearby island of Mehetia.',
                 ' The'
                 ' Tahiti rail appears to have been closely related to, and perhaps derived from, the buff-banded rail,'
                 ' and has also been historically confused with the Tongan subspecies of that bird.',
                 'The Tahiti rail was 9 inches (23 centimetres) long, and its colouration was unusual for a rail.'
                 ' The underparts, throat, and eyebrow-like supercilium were white, and the upper parts were black '
                 'with white dots and bands.',
                 ' The nape (or hind neck) was ferruginous (rust-coloured), the breast was '
                 'grey, and it had a black band across the lower throat. ',
                 'The bill and iris were red, and the legs '
                 'were fleshy pink. ',
                 'The Tahiti rail was supposedly flightless, and nested on the ground. It is said '
                 'to have been seen in open areas, marshes, and in coconut plantations. ',
                 'Its diet appears to have '
                 'consisted mainly of insects and occasionally copra (coconut meat). The extinction of the Tahiti '
                 'rail was probably due to predation by humans and introduced cats and rats.',
                 ' It appears to have become'
                 ' extinct some time after 1844 on Tahiti, and perhaps as late as the 1930s on Mehetia.'
                 ]
    try:
        vectorizer = CountVectorizer(encoding='utf-8', lowercase=True, ngram_range=(1, 1), min_df=arg)
        X = vectorizer.fit_transform(raw_documents=documents)
        return len(vectorizer.vocabulary_) - 14.0
    except ValueError:
        return float('inf')


if __name__ == '__main__':
    g = newton(f, x0=17.0)
    print(g)
    print(f(g))

    g = newton(get_count, x0=4)
    print(g)
