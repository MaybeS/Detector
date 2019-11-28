import requests

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'
}

def download(url):
    try:
        contents = requests.get(url, headers=headers, stream=True)
        response = contents.raw

    except Exception as e:
        response = None
        raise e

    finally:
        return response
