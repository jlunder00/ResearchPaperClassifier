import pydoi
import json
from urllib.parse import urlparse

def get_items(fname, start,  n):
    with open(fname, 'r') as fin:
        lines = fin.readlines()
        return [json.loads(lines[i]) for i in range(start, n, 1)]

def get_dois(items):
    dois = []
    for item in items:
        if item['doi'] != None:
            dois.append(item['doi'])
    return dois

def get_resolved_dois(dois):
    return [pydoi.resolve(doi) for doi in dois]

def get_urls(resolved_dois):
    return [item['values'][0]['data']['value'] for item in resolved_dois if 'values' in item.keys()]

def get_hostnames(urls):
    return [urlparse(item).hostname.casefold() for item in urls]

def load(fname):
    with open(fname, 'r') as fin:
        return json.loads(fin.read())

def get_beals_urls(l):
    return [item['url'] for item in l]

def check_exists_within(l1, l2):
    return [item in l2 for item in l1]
        

def main():
    beals_pub = load('data/beals_publishers.json')
    beals_journal = load('data/beals_standalone_journals.json')
    beals_pub_hostnames = get_hostnames(get_urls(beals_pub))
    beals_journal_hostnames = get_hostnames(get_urls(beals_journal))

    start = 0
    end = 0
    for i in range(100, 5000, 100):
        start = end
        end = end + i
        items = get_items("data/arxiv-metadata-oai-snapshot.json", start, end)
        dois = get_dois(items)
        resolved_dois = get_resolved_dois(dois)
        urls = get_urls(resolved_dois)
        hostnames = get_hostnames(urls)
        hostnames_in_beals_pub = check_exists_within(hostnames, beals_pub_hostnames)
        hostnames_in_beals_journal = check_exists_within(hostnames, beals_journal_hostnames)
        any_match_pub = True in hostnames_in_beals_pub
        print('any match in pub: '+str(any_match_pub))
        any_match_journal = True in hostnames_in_beals_journal
        print('any match in journal: '+str(any_match_journal))
    print("done")

    


if __name__ == "__main__":
    main()


