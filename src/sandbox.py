import collections

# a module for defining your own datatype

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

Website = collections.namedtuple('Website', ['name', 'url', 'founder'])

websites = [
    ('Sohu', 'http://www.google.com/', u'张朝阳'),
    ('Sina', 'http://www.sina.com.cn/', u'王志东'),
    ('163', 'http://www.163.com/', u'丁磊')
]

for website in websites:
    website = Website._make(website)
    print (website)