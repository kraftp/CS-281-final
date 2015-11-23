import flickrapi
import json
import requests
import os

with open("./settings.config", "rb") as settings_file:
    settings = json.load(settings_file)
    flickr = flickrapi.FlickrAPI(settings["KEY"], settings["SECRET"], format='parsed-json')

def get_images(tags, total):
    # NOTE: total is multiplied by 100
    to_return = []
    for page in xrange(1, 1 + total):
        response = flickr.photos.search(text=tags, per_page=100, page=page, sort='relevance')
        for photo in response["photos"]["photo"]:
            url = "https://farm{}.staticflickr.com/{}/{}_{}.jpg".format(photo["farm"], photo["server"], photo["id"], photo["secret"])
            to_return.append(url)
    return to_return

# Create postive cases
train_file_dir = "./train/"
test_file_dir = "./test/"

rotate = 7
i = 0

for tag, filetag, filenum in [("dog", "dog", 5), ("-dog", "notdog", 10)]:
    for image in get_images(tag, filenum):
        output_file_name = filetag + "_" + str(i) + '.jpg'
        output_file_dir = train_file_dir
        if i % rotate == 0:
            output_file_dir = test_file_dir
        with open(os.path.join(output_file_dir, output_file_name), 'wb') as output_file:
            output_file.write(requests.get(image).content)
        i += 1
