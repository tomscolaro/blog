---
layout: post
title:  "Taking Free Data"
author: Tom
categories: [ Articles, Python,  utilities]
image: assets/images/web-scraper-article.jpg
featured: true
project: false
---


IBIS World is a data provider that researches markets and current economic trends. It helps give marketers and entrepreneurs understand future outlook for a specific industry. While IBIS does have commercial solutions for more specific data, they offer a lot of useful information for free.

The goal of this script is to capture that data in a repeatable, scalable way.It uses web browser automation using Selenium. Selenium loans itself to a ton of flexibility, one of the applications being web scraping. This is especially powerful when web applications use JavaScript functions to generate content. As you can execute those scripts with selenium as well.

Let's dive in.



Our first step is to define and import our modules. This script is pretty basic, so it is only using base pandas (for writing our data into convenient data frames) and selenium which will do most of the navigation and extraction of values.


```python
import pandas as pd
import selenium

```

We define our web driver location on our local machine. Selenium uses these web drivers to execute the commands. I personally like to use Firefox's Geckodriver, but drivers for Chrome exist as well.  <INSERT LINKS FOR OTHER DRIVERS> If you do not already have a web driver installed on your computer (or on a selenium remote server), you will need to do so to use Selenium.


```python
driverLocation = "C:\CODE_PATHS\PY\geckodriver.exe"
browser = selenium.webdriver.Firefox(executable_path=driverLocation)
```

.get is a method in the browser class uses to execute a an HTTP/HTTPS link. This link takes me as far into the websites directory as I can, otherwise I would need additional steps of traversing the site to where I ultimately want to go.


```python
browser.get('https://www.ibisworld.com/industry-trends/market-research-reports/manufacturing/')
```

The function .find_elements_by_xpath() method allows the browser to find specific tags, classes and ids in an HTML page. Below, we can see that I am find all 'a' tags with HREF links in them. I want to iterate through these later, so I created a list I could append to from the loop. Then, I want to create a data frame, filter them down to the ones relevant to manufacturing and then return the list of strings to be navigated to later.


```python
elems = browser.find_elements_by_xpath("//a[@href]")
links = []
for elem in elems:
    links.append(elem.get_attribute("href"))

links = pd.DataFrame(links)
links.columns = ['Links']    

links=links[links['Links'].str.contains('industry-trends/market-research-reports/manufacturing')]
links = list(links['Links'].values)

```

All of the data that we need are in consistent tables. To convert into a workable format, it take a little work. As we have done before, we launch the gecko driver for each link that had previously been generated. The function finds the table and creates it as the <data> variable. It splits all the data into a list, and then the index of that list is used to build a Pandas data frame.  


```python
def table_getter(links_list):
    import time
    driverLocation = "C:\CODE_PATHS\PY\geckodriver.exe"
    browser = webdriver.Firefox(executable_path=driverLocation)

    mainData = pd.DataFrame()

    for i in links_list:
        browser.get(str(i))
        time.sleep(10)
        data = browser.find_element_by_class_name('stat-table').text
        dataName = browser.find_element_by_tag_name('h1').text
        data = data.split("\n")
        data.append(dataName)
        singleData = pd.DataFrame([[data[26],data[4],data[5],data[8],data[9],data[12],data[13],
                              data[17],data[18],data[19],data[20],data[21],data[22],
                                 data[23],data[24],data[25]]],
                columns = ['INDUSTRY', 'ANNUAL_REVENUE', 'COUNT_BUSINESSES',
                          'REVENUE_GROWTH_SINCE_2013','COUNT_BUSINESS_GROWTH_SINCE_2013',
                          'REVENUE_FORECAST_TO_2023', 'COUNT_BUSINESS_FORECAST_TO_2023', 'EMPLOYMENT',
                          'IMPORTS', 'EXPORTS', 'EMPLOYMENT_GROWTH_SINCE_2013', 'IMPORT_GROWTH_SINCE_2013',
                          'EXPORT_GROWTH_SINCE_2013', 'EMPLOYMENT_FORECAST_TO_2023', 'IMPORT_FORECAST_TO_2023',
                          'EXPORT_FORECAST_TO_2023'])
        mainData = pd.concat([mainData, singleData])
        data =[]
    mainData.reset_index()
    return mainData
```


```python
final = table_getter(links)
final.reset_index(inplace = True )
```


<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>INDUSTRY</th>
      <th>ANNUAL_REVENUE</th>
      <th>COUNT_BUSINESSES</th>
      <th>REVENUE_GROWTH_SINCE_2013</th>

    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Manufacturing - US Market Research Report</td>
      <td>$6,177.2 BN</td>
      <td>614,884</td>
      <td>-0.5 %</td>


    </tr>
    <tr>
      <th>0</th>
      <td>Apparel Manufacturing - US Market Research Report</td>
      <td>$6,177.2 BN</td>
      <td>614,884</td>
      <td>-0.5 %</td>    
    </tr>
    <tr>
      <th>0</th>
      <td>Beverage and Tobacco Product Manufacturing - U...</td>
      <td>$6,177.2 BN</td>
      <td>614,884</td>
      <td>-0.5 %</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Printing and Related Support Activities - US M...</td>
      <td>$6,177.2 BN</td>
      <td>614,884</td>
      <td>-0.5 %</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Textile Manufacturing - US Market Research Report</td>
      <td>$6,177.2 BN</td>
      <td>614,884</td>
      <td>-0.5 %</td>
    </tr>

  </tbody>
</table>






From this point, the data can be store in a SQL database or as a CSV file. It can be as regressors for predictive models or even for general reporting KPIs.

 Web scraping is a very lucrative process. As a data scientist, we often have the tools to perform an analysis, but not necessarily the data. Data like this is just begging to be harvested to create something cool. I hope you can apply this to your own work flow. If you have any questions, please feel free to reach out.
