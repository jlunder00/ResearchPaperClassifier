from scholarly import ProxyGenerator, scholarly
import json
from urllib.parse import urlparse
import signal



def handler(signum, frame):
    raise Exception("no more")


def get_items(fname, start,  n):

    with open(fname, 'r') as fin:
        lines = fin.readlines()
        items = []
        for i in range(start, n, 1):
            try:
                items.append(json.loads(lines[i]))
            except:
                return items
        return items

def get_titles(items):
    titles = []
    for item in items:
        if item['title'] != None:
            titles.append(item['title'])
    return titles

def get_num_citations(titles):
    print(titles)
    num_citations = []
    for title in titles:
        # try:
        num_citations.append(next(scholarly.search_pubs(title.replace('\n ', '')))['num_citations'])
        # except:
            # num_citations.append(None)
            
    return num_citations


def add_num_citations(items, num_citations):
    for i in range(len(items)):
        items[i]['num_citations'] = num_citations[i]
    return items

def save(fname, items, first, last):
    if first:
        with open(fname, 'w') as f:
            f.write('[\n')
            f.write(json.dumps(items)+',\n')
    elif last:
        with open(fname, 'a') as f:
            f.write(json.dumps(items)+'\n]')
    else:
        with open(fname, 'a') as f:
            f.write(json.dumps(items)+',\n')

if __name__ == '__main__':
    pg = ProxyGenerator()
    pg.FreeProxies()
    scholarly.use_proxy(pg)
    
    start = 0
    end = 2
    size = 1
    for i in range(size, 5000, size):
        start = end
        end = end + i
        items = get_items('data/arxiv-metadata-oai-snapshot.json', start, end)
        titles = get_titles(items)
        num_citations = get_num_citations(titles)
        items = add_num_citations(items, num_citations)
        save('data/with_num_citations', items, first = start==0, last=len(items) < size)
        
        


