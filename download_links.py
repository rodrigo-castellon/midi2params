# download a bunch of files from the user study into the github
# repo to avoid weird SSL cert (?) errors

import os
import urllib.request

link_body = 'https://nlp.stanford.edu/data/cdonahue/wavegenie_userstudy/v2/'
clip_nums = [
    8,
    10,
    12,
    16,
    17
]

clip_types = ['obligation', 'shoulder', 'grandmother', 'art', 'variant', 'bubble', 'drain']

save_path = 'data2'

for num in clip_nums:
    for tp in clip_types:
        pointer = os.path.join(tp, str(num).zfill(2) + '.wav')
        link = os.path.join(link_body, pointer)
        print(link)
        local_link = os.path.join(save_path, pointer)
        local_dir = os.path.split(local_link)[0]
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        urllib.request.urlretrieve(link, local_link)

