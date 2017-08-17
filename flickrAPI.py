#Cancemi Damiano - W82000075

import flickrapi
import urllib, json
import os, shutil, sys
from time import time

api_key = 'cacb93cdc983a7b7169b9aa74076bdc9'
api_secret = '2cf0746a3e52624a'

#api_key = 'fb64d4583a04a5e52cbf5fa757a197a6'
#api_secret = 'a0ae41dbbd2d3185'

flickr = flickrapi.FlickrAPI(api_key, api_secret)

# BUILD PATH & FOLDER
path = os.getenv("HOME")+"/PycharmProjects/SMM/Progetto/Dataset/"
'''
if os.path.isdir(path):
	shutil.rmtree(path)
	os.mkdir(path)
'''
classes = ["bird", "mammal"]

per_page = 490
pages = 2

print "---- DOWNLOADING FLICKR DATASET (%d images per class) ----" % (per_page*pages)
for hashtag in classes:
	start_time = time()

	if not os.path.isdir(path):
		os.mkdir(path + hashtag, 0775)

	for count_page in range(1, pages+1):
		print "\n", '> Processing: flickr.photos_search(page=%d, sort="relevance", per_page=%d, text="%s", tags="animal")' % (count_page, per_page, hashtag)
		f = flickr.photos_search(page=count_page, sort="relevance", per_page=per_page, text=hashtag, tags='animal', format="json", api_key=api_key, media="photos")
		urllist = []
		photoarray = json.loads(f)
		photoarray = (photoarray['photos'])
		photoarray = photoarray['photo']

		# get id from JSON
		count_file = 0
		while count_file < len(photoarray):
			filename = "picture_" + str(count_page) + "_" + str(count_file) + ".jpg"
			full_path = path + hashtag + "/" + filename
			if not (os.path.isfile(full_path)):
				uid = str(photoarray[count_file]['id'])
				server = str(photoarray[count_file]['server'])
				farm = str(photoarray[count_file]['farm'])
				secret = str(photoarray[count_file]['secret'])
				title = photoarray[count_file]['title']
				url = "https://farm"+farm+".staticflickr.com/"+server+"/"+uid+"_"+secret+".jpg"

				urllib.urlretrieve(url, full_path)
				print '[%s] -> [%s]' % (title, filename)

			count_file += 1

	end_time = time()
	elapsed_time = end_time - start_time
	print "> Class %s downloaded. Time: {0:0.2f} sec.".format(elapsed_time) % hashtag


