"""
Exports Issues from a specified repository to a CSV file

Uses basic authentication (Github username + password) to retrieve Issues
from a repository that username has access to. Supports Github API v3.
"""
import csv
import requests


GITHUB_USER = ''
GITHUB_PASSWORD = ''
REPO = 'neutronimaging/imagingsuite'  # format is username/repo
ISSUES_FOR_REPO_URL = 'https://api.github.com/repos/%s/issues' % REPO

def write_issues(response):
    "output a list of issues to csv"
    if not r.status_code == 200:
        raise Exception(r.status_code)
    for issue in r.json():
        labels = issue['labels']
        if (len(labels)!=0) :
            print(labels[0]['name'])
            titlestr = "GitHub Issue #{}: {}".format(issue['number'],issue['title'])
            bodystr = issue['body']
            bodystr.replace('\\r\\n',' ')
#        print(issue['number'], issue['title'].encode('utf-8'), issue['body'].encode('utf-8'), issue['created_at'], issue['updated_at'])
            csvout.writerow([ issue['number'], labels[0]['name'].encode('utf-8') ,titlestr.encode('utf-8'), bodystr.encode('utf-8'), issue['created_at'], issue['updated_at'] ])


print(ISSUES_FOR_REPO_URL)

r= requests.get(ISSUES_FOR_REPO_URL)

csvfile = '%s-issues.csv' % (REPO.replace('/', '-'))
csvout = csv.writer(open(csvfile, 'w'))
csvout.writerow(('id','Issue type' ,'Title', 'Body', 'Created At', 'Updated At'))
write_issues(r)

#more pages? examine the 'link' header returned
if 'link' in r.headers:
    pages = dict(
        [(rel[6:-1], url[url.index('<')+1:-1]) for url, rel in
            [link.split(';') for link in
                r.headers['link'].split(',')]])
    while 'last' in pages and 'next' in pages:
        r = requests.get(pages['next'])
        write_issues(r)
        if pages['next'] == pages['last']:
            break