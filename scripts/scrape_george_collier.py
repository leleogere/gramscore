import time

from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Set up the GeckoDriver service
service = Service(
    '/home/gerel/Téléchargements/geckodriver'
)  # Replace with the path to your GeckoDriver

# Initialize the Firefox WebDriver
driver = webdriver.Firefox(service=service)

# Open the webpage
driver.get('https://georgecolliermusic.com')

time.sleep(3)

try:
    # Find the element with class "grid"
    grid_element = driver.find_element(By.CLASS_NAME, 'grid')

    # Find all sub-elements with class "bg-card" within the grid element
    bg_card_elements = grid_element.find_elements(By.CLASS_NAME, 'bg-card')
    print(f"Found {len(bg_card_elements)} bg cards")

    # Iterate over each "bg-card" element
    for bg_card in bg_card_elements[107:]:
        # Find the nested element with class "text-lg" within the current "bg-card" element
        text_lg_element = bg_card.find_element(By.CLASS_NAME, 'text-lg')
        print(text_lg_element.text)

        # Click on the card
        # print("Clicking card")
        bg_card.click()
        # print("Card clicked")

        # Find the download button
        time.sleep(1)
        # print("Finding 'Download' button")

        try:
            WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable(
                    (By.XPATH, "(//*[contains(text(), 'Download')])[last()]")
                )
            ).click()
        except:
            print(f"Error downloading {text_lg_element.text}")
        # breakpoint()
        # print("'Download' button found")

        # Click on the "Download" button
        # print("Clicking 'Download' button")
        # download_button.click()
        # print("'Download' button clicked")

        # Wait for the "Close" button to be clickable and find it
        time.sleep(1)
        # print("Finding 'Close' button")
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[.//span[contains(text(), 'Close')]]"))
        ).click()
        # print("'Close' button found")

        # Click on the "Close" button to close the popup
        # print("Clicking 'Close' button")
        # close_button.click()
        # print("'Click' button clicked")
        time.sleep(1)


except Exception as e:
    print(e)


# Close the browser
finally:
    input()
    driver.quit()
