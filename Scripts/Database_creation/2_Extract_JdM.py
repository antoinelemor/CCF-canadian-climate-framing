import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.firefox import GeckoDriverManager
import pyautogui
import time

def enter_proxy_auth(username, password):
    # Attendre que l'invite d'authentification apparaisse
    time.sleep(2)  # Peut nécessiter un ajustement selon la vitesse de votre système
    pyautogui.typewrite(username)
    pyautogui.press('tab')
    pyautogui.typewrite(password)
    pyautogui.press('enter')

def scraper_texte_article(url, driver):
    try:
        driver.get(url)
        # Supposer que l'invite d'authentification proxy s'affiche à ce moment-là
        enter_proxy_auth('p1241479', 'Felicis19')  # Utilisez vos vraies informations d'authentification ici
        # Attendre que la page charge après l'authentification
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "docOcurrContainer")))
        soup = BeautifulSoup(driver.page_source, "html.parser")
        texte_article = soup.find("div", class_="docOcurrContainer").get_text(separator=" ", strip=True)
        return texte_article
    except Exception as e:
        print(f"Erreur lors du scraping de l'URL {url}: {e}")
        return "Erreur lors du scraping"

def main():
    chemin_csv = '/Users/antoine/Documents/GitHub/CLIMATE.FRAME/Data/Raw_data/Journal_de_Montreal/clean_JDM_data/clean_JDM_data.csv'
    df = pd.read_csv(chemin_csv)
    
    # Enlever l'option headless pour permettre à PyAutoGUI de voir la fenêtre du navigateur
    options = webdriver.FirefoxOptions()
    # options.add_argument("--headless")  # Retiré pour PyAutoGUI
    driver = webdriver.Firefox(service=Service(GeckoDriverManager().install()), options=options)
    
    try:
        for i, row in df.iterrows():
            df.at[i, 'Texte'] = scraper_texte_article(row['URL'], driver)
    finally:
        driver.quit()
    
    nouveau_chemin_csv = '/Users/antoine/Documents/GitHub/CLIMATE.FRAME/Data/Raw_data/Journal_de_Montreal/clean_JDM_data/articles_extraits.csv'
    df.to_csv(nouveau_chemin_csv, index=False, encoding='utf-8')
    print(f"Fichier CSV mis à jour avec succès. Chemin : {nouveau_chemin_csv}")

if __name__ == '__main__':
    main()
