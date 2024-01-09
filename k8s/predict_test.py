import requests

# server_url = 'http://127.0.0.1:9696/predict'
server_url = 'http://127.0.0.1:8080/predict'

# pizza
data = {'url': 'https://m.kafeteria.pl/shutterstock-84904912-9cb8cae338,730,0,0,0.jpg'}

# not a pizza
# data = {'url': 'https://cdn.galleries.smcloud.net/t/galleries/gf-S6yu-gm6t-A9ZJ_zapiekanka-dworcowa-664x442-nocrop.jpg'}

result = requests.post(server_url, json=data)
print(result)
