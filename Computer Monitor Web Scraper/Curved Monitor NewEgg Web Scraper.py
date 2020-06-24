"""
Goal
Build a web scraper to check for brands, prices, shipping status, ratings, reviewers, promotions and stock status for Curved Monitors on NewEgg under $1000""
"""
#----------------------------------------------------------------------------------------------------------
#import packages
#----------------------------------------------------------------------------------------------------------
#pip install bs4
import bs4
from bs4 import BeautifulSoup as soup
from urllib.request import urlopen

#----------------------------------------------------------------------------------------------------------
#read in data/ urls
#----------------------------------------------------------------------------------------------------------

url = 'https://www.newegg.com/p/pl?N=4018%204019%204814%20100898493%20601318855&d=curved%20monitor&PageSize=96'
#url = 'https://www.newegg.com/p/pl?d=ultrawide+monitor&cm_sp=KeywordRelated-_-curved%20monitor-_-ultrawide%20monitor-_-INFOCARD&N=4016'
webpage = urlopen(url)
page_html = webpage.read()
webpage.close()

#parse HTML
webpage_soup = soup(page_html, "html.parser")
webpage_soup.h1  #verify if the header is correct "Curved Monitor"

#create containers - essentially, HTML specific to each product on the webpage
containers = webpage_soup.findAll("div",{"class": "item-container"})   
len(containers)  #number of products on the webpage

#the first top 3 images are "recommendations" and not part of our search so we will delete them 
del containers[:4]  #deletes the recommendations on top of the page

#delete products without branding and product name - adjust this to exclude products of choice, I just wanted to make sure my products had a name and branding
final_containers = []
for container in containers:
    if container.findAll("div", {"class": "item-branding"})[0].text.strip() != '' and container.findAll("a", {"class": "item-title"})[0].text != '':
        final_containers.append(container)

len(final_containers)


#sample container/ product HTML - sample infomation
product_1_html = containers[0]
#find brand, product name, previous price, current price, rating, number of reviewers and promo description of a sample product
product_1_html.findAll("a", {"class": "item-brand"})[0].img["title"]  #brand
product_1_html.findAll("a", {"class": "item-title"})[0].text #product name
product_1_html.findAll("li", {"class": "price-was"})[0].span.text  #previous price
product_1_html.findAll("li", {"class": "price-current"})[0].text.strip()[0:7] #current price
product_1_html.findAll("a", {"class": "item-rating"})[0]["title"].replace("Rating + ", "")  #rating
product_1_html.findAll("a", {"class": "item-rating"})[0].span.text.replace("(", "").replace(")", "")  #number of reviewers
product_1_html.findAll("li", {"class": "price-ship"})[0].text.strip()  #shipping status
product_1_html.findAll("p", {"class": "item-promo"})[0].text #promo description

#----------------------------------------------------------------------------------------------------------
#create csv file for all products in containers
#----------------------------------------------------------------------------------------------------------
os.chdir("/Users/duanemadziva/Documents/_ Print (Hello World)/Learning Python/PythonVS/Data")
filename = "curved monitors newegg.csv"   
f = open(filename, "w")  #write to a file in current working directory

headers = "Brand, Name, Price, Shipping, Promotions, Rating, Reviewers\n"
f.write(headers)

for container in final_containers:
    brand = container.findAll("a", {"class": "item-brand"})[0].img["title"]
    product_name = container.findAll("a", {"class": "item-img"})[0].img["title"]
    current_price = container.findAll("li", {"class": "price-current"})[0].text.strip()[0:7]
    shipping = container.findAll("li", {"class": "price-ship"})[0].text.strip()
    promo = container.findAll("p", {"class": "item-promo"})[0].text
    rating = container.findAll("a", {"class": "item-rating"})[0]["title"].replace("Rating + ", "")
    number_reviewers = container.findAll("a", {"class": "item-rating"})[0].span.text.replace("(", "").replace(")", "")
    #print(rating)
    f.write(brand.replace(",", "") + "," + product_name.replace(",", "") + "," + current_price + "," + shipping + "," + promo.replace(",", "") + "," + rating + "," + number_reviewers + "\n")
f.close()  #always close the file, so you can actually open it 
os.chdir("/Users/duanemadziva/Documents/_ Print (Hello World)/Learning Python/PythonVS/Tech With Tim and Tutorials")










