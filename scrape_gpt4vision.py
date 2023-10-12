# coding=utf-8
import json
import time

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.edge.service import Service


def open_browser() -> None:
    options = Options()

    # https://msedgedriver.azureedge.net/117.0.2045.40/edgedriver_linux64.zip
    # options.add_argument("--headless")  # for running headlessly

    browser = webdriver.Chrome(options=options)
    browser.implicitly_wait(5)

    base_url = "https://chat.openai.com/"
    # session_id = browser.session_id
    # driver = webdriver.Remote(command_executor=base_url, options=options)
    # driver.close()
    # driver.session_id = session_id

    # base_url = "https://www.bing.com/search?q=Bing+AI&showconv=1&FORM=hpcodx"
    browser.get(base_url)
    time.sleep(5)

    # browser.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

    login_button = browser.find_element(By.XPATH, "//*[@id='__next']/div[1]/div[2]/div[1]/div/div/button[1]")
    login_button.click()
    time.sleep(5)

    with open("login_info.json", mode="r") as f:
        login_info = json.load(f)

    name_login = browser.find_element(By.ID, "username")
    name_login.send_keys(login_info["username"])
    name_login.send_keys(Keys.ENTER)
    time.sleep(5)

    password_login = browser.find_element(By.ID, "password")
    password_login.send_keys(login_info["password"])
    password_login.send_keys(Keys.ENTER)
    time.sleep(5)

    button_lets_go = browser.find_element(By.XPATH, "//*[@id='radix-:r1o:']/div[2]/div/div[4]/button/div")
    if button_lets_go:
        button_lets_go.click()
        time.sleep(5)

    button_gpt4 = browser.find_element(By.ID, "radix-:r1t:")
    button_gpt4.click()
    time.sleep(5)

    button_add_image = browser.find_element(By.XPATH, "//*[@id='__next']/div/div[2]/div[1]/div[2]/main/div[1]/div[2]/form/div/div[2]/div/div/span/button")
    button_add_image.click()
    time.sleep(5)

    print()


def main() -> None:
    open_browser()


if __name__ == '__main__':
    main()
