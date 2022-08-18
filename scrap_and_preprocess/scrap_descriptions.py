from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
import pandas as pd
from tqdm import tqdm
import time

# script for collecting job postings descriptions

postings_data = pd.read_csv('../job_postings.csv')
# this column was deleted from final version of the dataset
posting_hrefs = postings_data['posting_href'].to_list()

service = Service("../chromedriver.exe")
options = webdriver.ChromeOptions()
# options.add_argument("--headless")
driver = webdriver.Chrome(service=service, options=options)

driver.get('https://nofluffjobs.com/pl/warszawa?page=1&lang=en')
# accept cookies usage, so that other page elements become clickable
WebDriverWait(driver, 5).until(ec.element_to_be_clickable((By.ID, 'onetrust-accept-btn-handler'))).click()

postings_descrs = []
for i in tqdm(range(len(posting_hrefs))):
    driver.get(posting_hrefs[i])

    # some descriptions are too big, so script waits for read more button
    try:
        WebDriverWait(driver, 2).until(ec.element_to_be_clickable((By.XPATH,
                                                                   "//a[contains(@class, 'read-more')]"))).click()
    except TimeoutException:
        pass

    # some descriptions are published in Polish, so script checks if there is translation button
    translate_button = driver.find_elements(By.XPATH, "//button[contains(@class, 'text-primary')]")
    if len(translate_button) > 0:
        translate_button[0].click()
    time.sleep(1)

    # find description container and process it
    posting_descr = driver.find_elements(By.XPATH, "//p[@class='font-weight-normal']")
    if len(posting_descr) > 0:
        text = posting_descr[0].text
        text = text[0:text.rfind("\n")]
    else:
        text = ''
    postings_descrs.append(text)

postings_data['description'] = postings_descrs
postings_data.to_csv('job_postings.csv', index=False, encoding='utf-8-sig')

driver.quit()

