from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
import time

service = Service("chromedriver.exe")
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=service, options=options)

no_fluff_job_path = 'https://nofluffjobs.com/pl/warszawa?lang=en&page=1'

driver.get(no_fluff_job_path)

WebDriverWait(driver, 10).until(ec.element_to_be_clickable((By.ID, 'onetrust-accept-btn-handler'))).click()

postings = driver.find_elements(By.XPATH, "//a[contains(@class, 'posting-list-item')]")
print(postings[5].find_element(By.XPATH, "//h3[contains(@class, 'posting-title__position')]").text)
posting_href10 = postings[10].get_attribute('href')
postings[7].click()

try:
    WebDriverWait(driver, 2).until(ec.element_to_be_clickable((By.XPATH,
                                                            "//a[contains(@class, 'read-more')]"))).click()
except TimeoutException:
    pass
translate_button = driver.find_elements(By.XPATH, "//button[contains(@class, 'text-primary')]")
if len(translate_button) > 0:
    translate_button[0].click()
print(driver.find_element(By.XPATH, "//p[@class='font-weight-normal']").text)
posting_href10 += "?lang=eng"
print(posting_href10)
driver.get("https://nofluffjobs.com/pl/job/remote-java-engineer-link-group-warszawa-c4eylvqj?lang=eng")


# driver.quit()
