import json



def get_items(fname, start, n):

    with open(fname, 'r') as fin:
        lines = fin.readlines()
        items = []
        for i in range(start, n, 1):
            try:
                item = json.loads(lines[i])
                if 'abstract' in list(item.keys()) and 'title' in list(item.keys()):
                    items.append(item)
            except:
                return items
        return items

def save(fname, items, first, last):
    if first:
        with open(fname, 'w') as f:
            f.write('[\n')
            f.write(json.dumps(items, indent=4)+',\n')
    elif last:
        with open(fname, 'a') as f:
            f.write(json.dumps(items, indent=4)+'\n]')
    else:
        with open(fname, 'a') as f:
            f.write(json.dumps(items, indent=4)+',\n')

if __name__ == '__main__':
    start = 0
    end = 0
    size = 100
    for i in range(size, 1000000, size):
        start = end
        end = end + i
        items = get_items('data/arxiv-metadata-oai-snapshot.json', start, end)
        save('data/with_titles_and_abstract.json', items, first = start==0, last=len(items) < size)
        
        


 
