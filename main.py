# coding=utf-8
import json
import time

from selenium import webdriver
# from webdriver_manager.chrome import ChromeDriverManager
# from selenium.webdriver.chrome.options import Options
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.edge.service import Service


def open_browser() -> None:
    options = Options()

    # https://msedgedriver.azureedge.net/117.0.2045.40/edgedriver_linux64.zip
    # options.add_argument("--headless")  # for running headlessly

    exec_path = "/usr/bin/microsoft-edge"
    options.binary_location = exec_path

    driver_path = "/home/mark/Downloads/msedgedriver"
    service = Service(executable_path=driver_path)

    browser = webdriver.Edge(service=service, options=options)
    # browser = webdriver.Chrome(options=options)
    browser.implicitly_wait(5)

    base_url = "https://www.bing.com"
    # session_id = browser.session_id
    # driver = webdriver.Remote(command_executor=base_url, options=options)
    # driver.close()
    # driver.session_id = session_id

    # base_url = "https://www.bing.com/search?q=Bing+AI&showconv=1&FORM=hpcodx"
    browser.get(base_url)
    time.sleep(5)

    browser.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

    login_button = browser.find_element(By.ID, "id_s")
    login_button.click()
    time.sleep(5)

    with open("login_info.json", mode="r") as f:
        login_info = json.load(f)

    name_login = browser.find_element(By.ID, "i0116")
    name_login.send_keys(login_info["username"])
    name_login.send_keys(Keys.ENTER)
    time.sleep(5)

    password_login = browser.find_element(By.ID, "i0118")
    password_login.send_keys(login_info["password"])
    password_login.send_keys(Keys.ENTER)
    time.sleep(5)

    keep_logged_in_checkbox = browser.find_element(By.ID, "KmsiCheckboxField")
    keep_logged_in_checkbox.click()
    stay_logged_in_button = browser.find_element(By.ID, "idSIButton9")
    stay_logged_in_button.click()
    time.sleep(5)

    button_reject = browser.find_element(By.ID, "bnp_btn_reject")
    if button_reject:
        button_reject.click()

    print()


def main() -> None:
    open_browser()


if __name__ == '__main__':
    main()
